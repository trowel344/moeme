#!/usr/bin/env python3
"""emit_moe_gguf.py -- emit a standard qwen3moe GGUF from a converted model.

Layout matches surplus' native reader EXACTLY (model.rs): file-order dims,
expert-major blobs, F32/F16/Q8_0/Q4_0 packings per kernels.rs decode.

Sliced tiles are zero-padded to full width: with top_k=E, uniform routing
(zeros) and expert_scale=E the file is a mathematically exact dense
equivalent (validation emit). With trained router + top_k=k + scale=1 it is
the MoCM. No shared experts (reader rejects ffn_shared*).

Streaming: layout computed from geometry, header written first, weights
stream layer-by-layer (lazy safetensors reads). Peak RAM ~2GB.

  python3 tools/emit_moe_gguf.py --base Qwen/Qwen3-4B \
      --layers $(seq -s, 0 35) --experts 2 --top-k 2 --expert-scale 2.0 \
      --dtype f16 --out model.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

_GGUF_STRING, _GGUF_U32, _GGUF_F64 = 8, 4, 12
_DID = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}
_ALIGN = 32


def _pad(n: int, a: int = _ALIGN) -> int:
    return (a - n % a) % a


def _enc_str(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def _kv(kvs: list) -> bytearray:
    out = bytearray()
    for key, typ, val in kvs:
        out += _enc_str(key.encode())
        if typ == "s":
            out += struct.pack("<I", _GGUF_STRING) + _enc_str(val.encode())
        elif typ == "u":
            # u32 (not u64): llama.cpp rejects u64 arch keys, and surplus
            # native accepts u32 too. All our values fit 32 bits.
            assert 0 <= val < 2 ** 32, (key, val)
            out += struct.pack("<I", _GGUF_U32) + struct.pack("<I", val)
        elif typ == "f":
            # f32 (not f64): llama.cpp rejects f64 arch keys, surplus reads
            # both (kinds 6 and 12 in kernels.rs).
            out += struct.pack("<I", 6) + struct.pack("<f", val)
        elif typ == "b":
            out += struct.pack("<I", 7) + struct.pack("<B", 1 if val else 0)
        elif typ == "arr-s":
            out += struct.pack("<I", 9) + struct.pack("<I", 8)
            out += struct.pack("<Q", len(val))
            for s in val:
                out += _enc_str(s.encode("utf-8", "replace"))
        elif typ == "arr-f":
            import array as _arr
            out += struct.pack("<I", 9) + struct.pack("<I", 6)
            out += struct.pack("<Q", len(val))
            out += _arr.array("f", val).tobytes()
        elif typ == "arr-i":
            import array as _arr
            out += struct.pack("<I", 9) + struct.pack("<I", 4)
            out += struct.pack("<Q", len(val))
            out += _arr.array("i", val).tobytes()
        else:
            raise ValueError(typ)
    return out


def pack_q8(arr32) -> bytes:
    import numpy as np
    flat = np.asarray(arr32, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    scale = (amax / 127.0).astype(np.float16)
    qi = np.clip(np.round(flat / np.where(amax == 0, 1, amax) * 127.0),
                 -128, 127).astype(np.int8)
    out = np.empty((flat.shape[0], 34), dtype=np.uint8)
    out[:, :2] = np.frombuffer(scale.tobytes(), dtype=np.uint8).reshape(-1, 2)
    out[:, 2:] = qi.view(np.uint8)
    return out.tobytes()


def pack_q4(arr32) -> bytes:
    import numpy as np
    flat = np.asarray(arr32, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    scale = (amax / 8.0).astype(np.float16)
    qi = np.clip(np.round(flat / np.where(amax == 0, 1, amax) * 8.0) + 8.0,
                 0, 15).astype(np.uint8)
    out = np.empty((flat.shape[0], 18), dtype=np.uint8)
    out[:, :2] = np.frombuffer(scale.tobytes(), dtype=np.uint8).reshape(-1, 2)
    out[:, 2:] = (qi[:, ::2] | (qi[:, 1::2] << 4))
    return out.tobytes()


def encode_blob(t: torch.Tensor, dtype: str, force_f32: bool = False) -> tuple[bytes, int]:
    import numpy as np
    if force_f32:
        dtype = "f32"
    if dtype in ("q8_0", "q4_0"):
        assert t.shape[-1] % 32 == 0, "quant packing needs mult-of-32 rows"
    if dtype == "f32":
        return t.float().numpy().astype(np.float32).tobytes(), _DID["f32"]
    if dtype == "f16":
        return t.float().numpy().astype(np.float16).tobytes(), _DID["f16"]
    if dtype == "q8_0":
        return pack_q8(t.float().numpy()), _DID["q8_0"]
    if dtype == "q4_0":
        return pack_q4(t.float().numpy()), _DID["q4_0"]
    raise ValueError(dtype)


def blob_nbytes(numel: int, contig: int, dtype: str) -> int:
    if dtype == "f32":
        return numel * 4
    if dtype == "f16":
        return numel * 2
    # quantized packings block along the contiguous axis (mult of 32 asserted)
    per = {"q8_0": 34, "q4_0": 18}[dtype]
    return numel // 32 * per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert-scale", type=float, default=1.0)
    ap.add_argument("--dtype", default="f16",
                    choices=["f32", "f16", "q8_0", "q4_0"])
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--verify-only", default=None,
                    help="only verify an existing file, do not emit")
    args = ap.parse_args()
    if args.verify_only:
        verify_gguf(args.verify_only, args.dtype)
        return 0

    from transformers import AutoConfig, AutoTokenizer  # noqa
    from safetensors import safe_open  # noqa
    from huggingface_hub import snapshot_download  # noqa
    from moeme.branch import BranchConfig, BranchMoEFFN  # noqa

    torch.manual_seed(args.seed)
    snap = snapshot_download(args.base,
                             allow_patterns=["*.safetensors", "*.json"])
    cfg = AutoConfig.from_pretrained(args.base)
    tok = AutoTokenizer.from_pretrained(args.base)
    H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    heads, kvh = cfg.num_attention_heads, cfg.num_key_value_heads
    # Qwen3 head_dim is 128 regardless of hidden/heads (NOT H//heads);
    # surplus derives head_dim from the q_norm tensor, so this must agree.
    hd = int(getattr(cfg, "head_dim", H // heads))
    V = cfg.vocab_size
    layers = [int(v) for v in (args.layers.split(",") if args.layers
                               else ["28", "29", "30", "31"])]
    E = args.experts
    assert I % E == 0, "tile init needs I divisible by E"
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    files = {}

    def T(name):
        fn = wmap[name]
        if fn not in files:
            files[fn] = safe_open(os.path.join(snap, fn), framework="pt")
        return files[fn].get_tensor(name).float()

    pre = "model.layers" if "model.layers.0.mlp.gate_proj.weight" in wmap \
        else "model.language_model.layers"
    ckpt = torch.load(args.ckpt, map_location="cpu",
                       weights_only=False) if args.ckpt else None

    kvs = [
        ("general.architecture", "s", "qwen3moe"),
        ("general.name", "s", f"MoCM-{os.path.basename(args.base)}-E{E}k{args.top_k}"),
        ("qwen3moe.embedding_length", "u", H),
        ("qwen3moe.block_count", "u", L),
        ("qwen3moe.attention.head_count", "u", heads),
        ("qwen3moe.attention.head_count_kv", "u", kvh),
        # head_dim is NOT hidden/heads on Qwen3 (128 vs 80 here): declare it
        # explicitly or llama.cpp defaults key_length wrong and rejects the
        # tensors. Surplus cross-checks these against q_norm when present.
        ("qwen3moe.attention.key_length", "u", hd),
        ("qwen3moe.attention.value_length", "u", hd),
        ("qwen3moe.rope.dimension_count", "u", hd),
        ("qwen3moe.expert_count", "u", E),
        ("qwen3moe.expert_used_count", "u", args.top_k),
        ("qwen3moe.expert_feed_forward_length", "u", I),
        ("qwen3moe.context_length", "u", int(cfg.max_position_embeddings)),
        ("qwen3moe.attention.layer_norm_rms_epsilon", "f", float(cfg.rms_norm_eps)),
        ("qwen3moe.rope.freq_base", "f", float(getattr(cfg, "rope_theta", 1000000.0))),
        ("qwen3moe.expert_weights_scale", "f", float(args.expert_scale)),
    ]
    # (name, dims_file_order, numel)
    specs = [("token_embd.weight", [H, V], H * V)]
    if not cfg.tie_word_embeddings:
        specs.append(("output.weight", [H, V], H * V))
    specs.append(("output_norm.weight", [H], H))
    A2 = [("attn_q.weight", heads * hd),
          ("attn_k.weight", kvh * hd),
          ("attn_v.weight", kvh * hd),
          ("attn_output.weight", heads * hd)]
    A1 = ["attn_q_norm.weight", "attn_k_norm.weight",
          "attn_norm.weight", "ffn_norm.weight"]
    for i in range(L):
        for nm, rows in A2:
            specs.append((f"blk.{i}.{nm}", [H, rows], H * rows))
    # rebuild norm specs cleanly (q/k norms are [head_dim], layer norms [H])
    specs = [s for s in specs if not s[0].startswith("blk.")]
    for i in range(L):
        for nm, rows in A2:
            if nm == "attn_output.weight":
                # o_proj raw is (H, heads*hd): contiguous axis heads*hd
                specs.append((f"blk.{i}.{nm}", [heads * hd, H],
                              (heads * hd) * H))
            else:
                specs.append((f"blk.{i}.{nm}", [H, rows], H * rows))
        specs.append((f"blk.{i}.attn_q_norm.weight", [hd], hd))
        specs.append((f"blk.{i}.attn_k_norm.weight", [hd], hd))
        specs.append((f"blk.{i}.attn_norm.weight", [H], H))
        specs.append((f"blk.{i}.ffn_norm.weight", [H], H))
        specs.append((f"blk.{i}.ffn_gate_inp.weight", [H, E], H * E))
        specs.append((f"blk.{i}.ffn_gate_exps.weight", [H, I, E], H * I * E))
        specs.append((f"blk.{i}.ffn_up_exps.weight", [H, I, E], H * I * E))
        specs.append((f"blk.{i}.ffn_down_exps.weight", [I, H, E], I * H * E))

    did = {"f32": _DID["f32"], "f16": _DID["f16"], "q8_0": _DID["q8_0"],
           "q4_0": _DID["q4_0"]}[args.dtype]
    # ---- tokenizer (llama.cpp needs this to serve chat) ---
    tj = json.load(open(os.path.join(snap, "tokenizer.json")))
    tcfg = {}
    try:
        tcfg = json.load(open(os.path.join(snap, "tokenizer_config.json")))
    except OSError:
        pass
    vocab = tj["model"].get("vocab", {})
    inv = [""] * V
    for piece, i in vocab.items():
        if 0 <= i < V:
            inv[i] = piece
    for i, p in enumerate(inv):
        if not p:
            inv[i] = "<|reserved_%d|>" % i
    merges = tj["model"].get("merges", [])
    # HF stores BPE merges as [a, b] pairs; GGUF wants "a b" strings.
    merges = [" ".join(p) if isinstance(p, (list, tuple)) else p
              for p in merges]
    types = [1] * V
    special = {}
    for role in ("bos_token", "eos_token", "unk_token", "pad_token"):
        v = tcfg.get(role)
        if isinstance(v, dict):
            v = v.get("id")
        if isinstance(v, str):
            for a in tj.get("added_tokens", []):
                if a.get("content") == v:
                    v = a.get("id")
                    break
        if isinstance(v, int) and 0 <= v < V:
            special[v] = role
            types[v] = 2 if role == "unk_token" else 3
    kvs += [
        ("tokenizer.ggml.model", "s", "gpt2"),
        ("tokenizer.ggml.tokens", "arr-s", inv),
        ("tokenizer.ggml.scores", "arr-f", [0.0] * V),
        ("tokenizer.ggml.token_type", "arr-i", types),
        ("tokenizer.ggml.merges", "arr-s", merges),
        ("tokenizer.ggml.bos_token_id", "u",
         next((i for i, r in special.items() if r == "bos_token"), 151643)),
        ("tokenizer.ggml.eos_token_id", "u",
         next((i for i, r in special.items() if r == "eos_token"), 151645)),
        ("tokenizer.ggml.add_bos_token", "b",
         bool(tcfg.get("add_bos_token", False))),
        ("tokenizer.ggml.add_eos_token", "b",
         bool(tcfg.get("add_eos_token", False))),
    ]
    chat_tmpl = tcfg.get("chat_template")
    if isinstance(chat_tmpl, str) and chat_tmpl:
        kvs.append(("tokenizer.chat_template", "s", chat_tmpl))

    # 1D norms + router always F32 (elementwise binary ops cannot mix
    # f32 activations with f16 weights; negligible bytes).
    def eff_dtype(name):
        if name.endswith(("norm.weight", "ffn_gate_inp.weight")):
            return "f32"
        return args.dtype

    # byte size per tensor from contiguous length
    sized = []
    for (n, d, _ne) in specs:
        contig = d[0]
        rest = 1
        for v in d[1:]:
            rest *= v
        sized.append((n, d, blob_nbytes(contig * rest, contig, eff_dtype(n))))

    kvb = _kv(kvs)
    recs = bytearray()
    off = 0
    for (n, d, nb) in sized:
        rec = _enc_str(n.encode()) + struct.pack("<I", len(d))
        rec += struct.pack("<%dQ" % len(d), *d)
        rec += struct.pack("<I", {"f32": _DID["f32"], "f16": _DID["f16"],
                                   "q8_0": _DID["q8_0"],
                                   "q4_0": _DID["q4_0"]}[eff_dtype(n)]) + struct.pack("<Q", off)
        recs += rec
        off += nb + _pad(nb)
    header = struct.pack("<4sIQ", b"GGUF", 3, len(sized))
    header += struct.pack("<Q", len(kvs)) + kvb + recs
    header += b"\x00" * _pad(len(header))

    fh = open(args.out, "wb")
    fh.write(header)

    def put(name, blob_t, dims):
        ff = name.endswith(("norm.weight", "ffn_gate_inp.weight"))
        blob, _ = encode_blob(blob_t, args.dtype, force_f32=ff)
        fh.write(blob)
        fh.write(b"\x00" * _pad(len(blob)))

    # embeddings / head / norm (file order == raw HF layout transposed-free:
    # dims[0] contiguous, so write raw (vocab,hidden) etc. directly)
    # NOTE: cfg.vocab_size is padded (151936) beyond the tokenizer (151643);
    # pad the embedding rows so dims and bytes agree.
    _emb = T("model.embed_tokens.weight")
    if _emb.shape[0] < V:
        _emb = torch.cat([_emb, torch.zeros(V - _emb.shape[0], H)])
    put("token_embd.weight", _emb, [H, V])
    if not cfg.tie_word_embeddings:
        put("output.weight", T("lm_head.weight"), [H, V])
    put("output_norm.weight", T("model.norm.weight"), [H])

    for i in range(L):
        bpre = f"{pre}.{i}"
        is_moe = i in layers
        # attention + norms stream straight through (raw layout)
        put(f"blk.{i}.attn_q.weight", T(f"{bpre}.self_attn.q_proj.weight"),
            [H, heads * hd])
        put(f"blk.{i}.attn_k.weight", T(f"{bpre}.self_attn.k_proj.weight"),
            [H, kvh * hd])
        put(f"blk.{i}.attn_v.weight", T(f"{bpre}.self_attn.v_proj.weight"),
            [H, kvh * hd])
        put(f"blk.{i}.attn_output.weight", T(f"{bpre}.self_attn.o_proj.weight"),
            [heads * hd, H])
        put(f"blk.{i}.attn_q_norm.weight", T(f"{bpre}.self_attn.q_norm.weight"),
            [hd])
        put(f"blk.{i}.attn_k_norm.weight", T(f"{bpre}.self_attn.k_norm.weight"),
            [hd])
        put(f"blk.{i}.attn_norm.weight", T(f"{bpre}.input_layernorm.weight"), [H])
        put(f"blk.{i}.ffn_norm.weight", T(f"{bpre}.post_attention_layernorm.weight"),
            [H])
        if is_moe:
            if ckpt:
                st = ckpt["moes"][str(i)]
                G, U, D = st["gate"].float(), st["up"].float(), st["down"].float()
                R = st["router.weight"].float()
            else:
                m = BranchMoEFFN(
                    T(f"{bpre}.mlp.gate_proj.weight"),
                    T(f"{bpre}.mlp.up_proj.weight"),
                    T(f"{bpre}.mlp.down_proj.weight"),
                    BranchConfig(n_experts=E, top_k=args.top_k, init="tile",
                                 expert_units=I // E, combine="sum", seed=args.seed))
                G, U, D = (t.detach().float() for t in (m.gate, m.up, m.down))
                R = torch.zeros(E, H)
            Ue = I // E
            Gf = torch.zeros(E, I, H)
            Uf = torch.zeros(E, I, H)
            Df = torch.zeros(E, H, I)
            for e in range(E):
                a, z = e * Ue, (e + 1) * Ue
                Gf[e, a:z] = G[e] if G.shape[1] == Ue else G[e, a:z]
                Uf[e, a:z] = U[e] if U.shape[1] == Ue else U[e, a:z]
                Df[e, :, a:z] = D[e] if D.shape[2] == Ue else D[e][:, a:z]
            put(f"blk.{i}.ffn_gate_inp.weight", R.contiguous(), [H, E])
            put(f"blk.{i}.ffn_gate_exps.weight", Gf.contiguous(), [H, I, E])
            put(f"blk.{i}.ffn_up_exps.weight", Uf.contiguous(), [H, I, E])
            # down checked as [ff, hidden, E]: per expert an (H, I) row-major
            # block -- Df already is (E, H, I), NO transpose.
            put(f"blk.{i}.ffn_down_exps.weight", Df.contiguous(), [I, H, E])
        else:
            # dense layer as E identical full replicas: with renormalized
            # weights every top-k set reproduces dense exactly.
            g = T(f"{bpre}.mlp.gate_proj.weight")
            u = T(f"{bpre}.mlp.up_proj.weight")
            d = T(f"{bpre}.mlp.down_proj.weight")
            put(f"blk.{i}.ffn_gate_inp.weight", torch.zeros(E, H), [H, E])
            put(f"blk.{i}.ffn_gate_exps.weight", g.unsqueeze(0).expand(E, -1, -1).contiguous(),
                [H, I, E])
            put(f"blk.{i}.ffn_up_exps.weight", u.unsqueeze(0).expand(E, -1, -1).contiguous(),
                [H, I, E])
            put(f"blk.{i}.ffn_down_exps.weight", d.unsqueeze(0).expand(E, -1, -1).contiguous(),
                [I, H, E])
        # free per-layer temporaries (safetensors lazy pages stay mapped)
        if i % 6 == 5:
            fh.flush()
    fh.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")
    print("tokenizer vocab:", tok.vocab_size, "tie:", cfg.tie_word_embeddings)

    # self-verification (after freeing write pressure)
    import gc as _gc
    try:
        files.clear()
    except NameError:
        pass
    _gc.collect()
    geo = {"H": H, "V": V, "heads": heads, "hd": hd, "I": I, "E": E,
           "n_spec": len(sized)}
    verify_gguf(args.out, args.dtype, geometry=geo)
    return 0


def verify_gguf(path, dtype, geometry=None):
    # dims/exactness of every record + no overlaps
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        f.read(4)
        nt = struct.unpack("<Q", f.read(8))[0]
        nkv = struct.unpack("<Q", f.read(8))[0]

        def gs():
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln)

        def skip_value(t):
            if t == 8:
                f.read(struct.unpack("<Q", f.read(8))[0])
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                w = {8: None, 6: 4, 4: 4, 10: 8, 11: 8, 12: 8}.get(et)
                if w is None:  # nested string array: lens + bytes
                    for _ in range(n):
                        f.read(struct.unpack("<Q", f.read(8))[0])
                else:
                    f.read(n * w)
            elif t in (10, 11, 12):
                f.read(8)
            elif t == 7:
                f.read(1)
            else:
                f.read(4)

        for _ in range(nkv):
            k = gs()
            t = struct.unpack("<I", f.read(4))[0]
            skip_value(t)
        seen, prev_end, bad = {}, 0, 0
        per = {"f32": 128, "f16": 64, "q8_0": 34, "q4_0": 18}[dtype]

        def expect_bytes(d):
            n = 1
            for v in d:
                n *= v
            return n // 32 * per if dtype.startswith("q") else n * \
                (4 if dtype == "f32" else 2)
        for _ in range(nt):
            name = gs().decode()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            kind = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            exp = expect_bytes(dims)
            if off < prev_end:
                print(f"  OVERLAP {name}")
                bad += 1
            prev_end = off + exp
            seen[name] = dims
        import os as _os
        dstart = f.tell() + _pad(f.tell())
        fsize = _os.path.getsize(path)
    assert bad == 0, f"{bad} overlapping records"
    print(f"verified {len(seen)} tensor records, no overlaps")
    if geometry:
        H, V = geometry["H"], geometry["V"]
        heads, hd, I, E = (geometry["heads"], geometry["hd"],
                           geometry["I"], geometry["E"])
        assert seen["token_embd.weight"] == [H, V], seen["token_embd.weight"]
        assert seen["blk.0.attn_q.weight"] == [H, heads * hd]
        assert seen["blk.0.attn_output.weight"] == [heads * hd, H]
        assert seen["blk.0.attn_q_norm.weight"] == [hd], "q_norm record missing"
        assert seen["blk.0.ffn_gate_exps.weight"] == [H, I, E]
        assert seen["blk.0.ffn_down_exps.weight"] == [I, H, E]
        assert len(seen) == geometry["n_spec"], \
            f"records {len(seen)} != spec {geometry['n_spec']}"
        print("geometry + record count match")
    # file size must equal header + described payloads: catches blobs
    # written without records (they shift everything silently).
    # NOTE: padding per record recomputed from dims; any mismatch aborts.
    print(f"file {fsize/1e9:.2f} GB, payload end {dstart + prev_end:,} "
          f"(slack {fsize - dstart - prev_end} bytes padding)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
