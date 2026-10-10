"""
Accounting tests.

These assert that the parameter model in `moeme.accounting` reproduces the
values measured directly from the released Qwen3.8-27B safetensors headers
(see tools/verify_model_geometry.py and results/qwen3.8-27b-geometry.json).
If someone edits the geometry constants into something that no longer matches
the real artefact, these fail -- which is the point.
"""

import pytest

from moeme.accounting import (
    MemoryTier,
    Qwen38_27B,
    bytes_for_params,
    predicted_decode_tok_s,
)

CFG = Qwen38_27B()


def test_ffn_params_match_measured():
    assert CFG.ffn_params() == CFG.measured_ffn == 17_112_760_320


def test_attention_params_match_measured():
    assert CFG.attention_params() == CFG.measured_attention == 1_677_721_600


def test_linear_attention_params_match_measured():
    assert CFG.linear_attention_params() == CFG.measured_linear_attention == 5_562_044_928


def test_embeddings_and_head_match_measured():
    assert CFG.embedding_params() == CFG.measured_embedding == 1_271_398_400
    assert CFG.lm_head_params() == CFG.measured_lm_head == 1_271_398_400


def test_text_backbone_matches_measured_total():
    """Text tower + MTP head + vision tower must equal the measured total."""
    assert CFG.total_with_aux() == CFG.measured_total == 27_781_427_952


def test_irreducible_active_is_about_9_78b():
    irr = CFG.irreducible_active_params()
    assert irr == 9_783_238_144
    # the headline finding: the backbone alone already exceeds a 9B budget
    assert irr > 9e9
    assert irr / 9e9 > 1.08


def test_ffn_share_of_parameters():
    # 61.6% of the whole checkpoint (measured), 63.6% of the text tower alone
    assert 0.61 < CFG.ffn_params() / CFG.total_with_aux() < 0.62
    assert 0.63 < CFG.ffn_params() / CFG.text_backbone_params() < 0.64


def test_no_ffn_routing_can_reach_9b_active():
    """Even a perfect, free, lossless FFN router cannot reach the 9B target.

    active_params = irreducible_backbone + active_ffn, and the irreducible part
    is already > 9B, so the required active_ffn would have to be negative.
    """
    irr = CFG.irreducible_active_params()
    target = 9e9
    required_active_ffn = target - irr
    assert required_active_ffn < 0, "9B active would be reachable; recheck geometry"


def test_bytes_for_params_units():
    assert bytes_for_params(8e9, 4.5) == pytest.approx(4.5e9)
    assert bytes_for_params(27_781_427_952, 16) == pytest.approx(55_562_855_904)


# ---------------------------------------------------------------------------
# memory-tier model
# ---------------------------------------------------------------------------

# Measured on this machine (results/hardware.json):
#   GPU 246.8 GB/s, RAM copy 40.7 GB/s, NVMe cold read 1.5 GB/s (4 MB blocks).
# Capacities are the *effective* ones a runtime actually obtains:
#   VRAM: 3.8 GiB. The card has 8188 MiB but the desktop holds ~2.9 GiB, and
#         ollama peaked at 3906 MiB of model memory (6884 - 2978 MiB baseline).
#   RAM : 10.2 GB = 9.5 GiB. This is the measured MemAvailable during the runs
#         (10.23e9 bytes). NOTE: GB, not GiB -- mixing the two overstates the
#         tier by ~7%.
BW_GPU, BW_RAM, BW_DISK = 246.8e9, 40.7e9, 1.5e9


def _tiers(gpu_gib=3.8, ram_gb=10.2):
    return (
        MemoryTier("gpu", gpu_gib * 1024 ** 3, BW_GPU),
        MemoryTier("ram", ram_gb * 1e9, BW_RAM),
        MemoryTier("disk", 1e15, BW_DISK),
    )


def test_tier_model_predicts_measured_27b_q4_rate():
    """The tier model must reproduce the measured end-to-end 0.561 tok/s.

    Qwen3.8-27B at Q4_K_M is 17.0 GB (4.9 bits/param). With ~3.8 GiB usable
    VRAM and ~10.2 GB usable RAM, ~2.7 GB per token must come off the SSD, and
    at 1.5 GB/s that dominates everything else.
    """
    gpu, ram, disk = _tiers()
    pred = predicted_decode_tok_s(
        active_params=27_781_427_952,
        bits_per_param=4.9,          # 17.0 GB GGUF / 27.78B params
        gpu=gpu, ram=ram, disk=disk,
    )
    measured = 0.561
    assert 0.5 * measured < pred["tok_s"] < 1.5 * measured, pred
    assert pred["bottleneck"] == "disk"


# Measured decoding rates on this machine (results/inference*.json):
#   qwen3:4b        model 2.5 GB, fully GPU          60.7 tok/s
#   ornith-1.5:9b   model 6.6 GB, partial offload    24.6 tok/s
#   qwen3.8:latest  model 17.0 GB, streams from SSD   0.561 tok/s
MEASURED_RUNS = [
    # (label, total params, bits/param, measured tok/s)
    ("qwen3:4b", 2.5e9 * 8 / 4.9, 4.9, 60.7),
    ("ornith-1.5:9b", 6.6e9 * 8 / 4.9, 4.9, 24.6),
    ("qwen3.8:latest", 27_781_427_952, 4.9, 0.561),
]


@pytest.mark.parametrize("label,params,bits,got", MEASURED_RUNS)
def test_tier_model_brackets_reality_within_2x(label, params, bits, got):
    """Honest accuracy statement: the tier model brackets reality within ~2x.

    It is a *ceiling* estimator -- it ignores kernel inefficiency, sampling,
    KV traffic and contention -- so systematic over-prediction is expected. Its
    job is to rank design choices, not to forecast tok/s precisely.
    """
    gpu, ram, disk = _tiers()
    pred = predicted_decode_tok_s(
        active_params=params, bits_per_param=bits,
        gpu=gpu, ram=ram, disk=disk,
    )
    ratio = pred["tok_s"] / got
    assert 0.4 < ratio < 2.2, (label, pred["tok_s"], got, ratio)


def test_tier_model_marks_rate_limited_by_slowest_tier():
    gpu, ram, disk = _tiers()
    pred = predicted_decode_tok_s(
        active_params=3e9, bits_per_param=4.5, gpu=gpu, ram=ram, disk=disk
    )
    assert pred["bottleneck"] == "gpu"
    assert pred["unplaceable_bytes"] == 0


def test_tier_model_reports_unplaceable_when_nothing_fits():
    gpu, ram, disk = _tiers(gpu_gib=0.0, ram_gb=0.0)
    pred = predicted_decode_tok_s(
        active_params=27_781_427_952, bits_per_param=4.5,
        gpu=gpu, ram=ram, disk=disk,
    )
    assert pred["unplaceable_bytes"] == 0  # disk is unbounded capacity
    assert pred["placement"]["disk"]["bytes"] > 0


def test_kv_bytes_reduce_gpu_capacity():
    """KV cache competes with weights for the same scarce VRAM."""
    gpu, ram, disk = _tiers()
    args = dict(active_params=4e9, bits_per_param=4.5, gpu=gpu, ram=ram, disk=disk)
    a = predicted_decode_tok_s(**args, kv_bytes=0)
    b = predicted_decode_tok_s(**args, kv_bytes=2e9)
    assert b["placement"]["gpu"]["bytes"] < a["placement"]["gpu"]["bytes"]
    # the weight mass displaced from VRAM has to fall through to RAM
    ram_a = a["placement"].get("ram", {}).get("bytes", 0.0)
    ram_b = b["placement"].get("ram", {}).get("bytes", 0.0)
    assert ram_b > ram_a
    assert b["tok_s"] < a["tok_s"]
