"""
moeme.accounting -- parameter/FLOP accounting for the Qwen3.8-27B geometry.

Purpose: make the central feasibility claim *executable and testable* rather
than an assertion in prose. The claim is that Qwen3.8-27B's non-FFN text
backbone is already ~9.78B parameters and is active on every token, so a
"9B active per token" target cannot be reached without retraining the backbone
itself -- routing the FFN is not sufficient.

All shapes below are taken from the verified artefact
(https://huggingface.co/Qwen/Qwen3.8-27B config.json, plus the safetensors
headers read by tools/verify_model_geometry.py). The test suite asserts that
the numbers computed here reproduce the measured weight-side totals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen38_27B:
    """Verified geometry of the text tower of Qwen/Qwen3.8-27B (dense)."""

    hidden: int = 5120
    intermediate: int = 17408
    num_layers: int = 64
    vocab: int = 248320
    # 16 x (3 GatedDeltaNet layers + 1 full-attention layer)
    full_attention_layers: int = 16
    linear_attention_layers: int = 48
    # full attention (fused q+gate projection)
    attn_q_dim: int = 24 * 256          # heads x head_dim, doubled by the fused gate
    attn_kv_dim: int = 4 * 256
    attn_o_dim: int = 24 * 256
    # gated delta net
    gdn_qk_dim: int = 16 * 128
    gdn_v_dim: int = 48 * 128
    # MTP head and vision tower are present in the released checkpoint
    mtp_params_measured: int = 424_699_392
    vision_params_measured: int = 460_730_096

    # ---- measured reference values, for assertions ----
    measured_total: int = 27_781_427_952
    measured_ffn: int = 17_112_760_320
    measured_linear_attention: int = 5_562_044_928
    measured_attention: int = 1_677_721_600
    measured_embedding: int = 1_271_398_400
    measured_lm_head: int = 1_271_398_400

    # ----- per-component accounting -----
    def ffn_params(self) -> int:
        # swiGLU: gate, up, down
        per_layer = 3 * self.hidden * self.intermediate
        return per_layer * self.num_layers

    def attention_params(self) -> int:
        # q_proj is fused q+gate => 2x the query width
        per_layer = (
            self.hidden * (2 * self.attn_q_dim)
            + 2 * self.hidden * self.attn_kv_dim
            + self.attn_q_dim * self.hidden
        )
        return per_layer * self.full_attention_layers

    def linear_attention_params(self) -> int:
        # in_proj_qkv, in_proj_z, out_proj + small in_proj_a/b, conv1d, A_log, dt_bias
        per_layer = (
            self.hidden * (2 * self.gdn_qk_dim + self.gdn_v_dim)   # in_proj_qkv
            + self.hidden * self.gdn_v_dim                          # in_proj_z
            + self.gdn_v_dim * self.hidden                          # out_proj
            + 2 * self.hidden * self.gdn_num_heads                  # in_proj_a, in_proj_b
            + self.gdn_conv_dim * 4                                 # conv1d, kernel 4
            + 2 * self.gdn_num_heads                                # A_log, dt_bias (1 scalar/head)
        )
        return per_layer * self.linear_attention_layers

    def norms_params(self) -> int:
        """2 RMSNorms per layer (10,240 each) + the DeltaNet norms + final norm."""
        return self.measured_norms

    measured_norms: int = 674_816

    gdn_num_heads: int = 48

    @property
    def gdn_conv_dim(self) -> int:
        return 2 * self.gdn_qk_dim + self.gdn_v_dim

    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    def lm_head_params(self) -> int:
        return 0 if self.tie_embeddings else self.vocab * self.hidden

    tie_embeddings: bool = False

    # ----- the decision-relevant totals -----
    def irreducible_active_params(self) -> int:
        """Params active on *every* token even if the whole FFN were free.

        This is the floor for the active-parameter count of any model that keeps
        the Qwen3.8-27B backbone untouched. Routing the FFN cannot go below it,
        so it is the hard limit on how small "active per token" can get.
        """
        return (
            self.embedding_params()
            + self.lm_head_params()
            + self.attention_params()
            + self.linear_attention_params()
            + self.norms_params()
        )

    def text_backbone_params(self) -> int:
        """Text tower only: FFN + attention + embeddings + norms.

        Equals the released text-tower parameter count (verified against the
        safetensors headers: 26,895,998,464).
        """
        return (
            self.irreducible_active_params() + self.ffn_params()
        )

    def total_with_aux(self) -> int:
        return self.text_backbone_params() + self.mtp_params_measured \
            + self.vision_params_measured

    def report(self) -> dict:
        irr = self.irreducible_active_params()
        ffn = self.ffn_params()
        text = self.text_backbone_params()
        return {
            "ffn_params": ffn,
            "attention_params": self.attention_params(),
            "linear_attention_params": self.linear_attention_params(),
            "embedding_params": self.embedding_params(),
            "lm_head_params": self.lm_head_params(),
            "irreducible_active_params": irr,
            "text_backbone_params": text,
            "total_with_aux": self.total_with_aux(),
            "ffn_share_of_text": ffn / text,
            "irreducible_over_9b": irr / 9e9,
            "measured_total": self.measured_total,
            "measured_ffn": self.measured_ffn,
        }


# ---------------------------------------------------------------------------
# memory-tier sizing
# ---------------------------------------------------------------------------

def bytes_for_params(params: float, bits_per_param: float) -> float:
    """Bytes to store `params` at a given weight precision."""
    return params * bits_per_param / 8.0


class MemoryTier:
    """A residency tier with a measured bandwidth (bytes/second)."""

    def __init__(self, name: str, capacity_bytes: float, bandwidth_bps: float):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.bandwidth_bps = bandwidth_bps

    def seconds_for(self, nbytes: float) -> float:
        return nbytes / self.bandwidth_bps if self.bandwidth_bps > 0 else float("inf")


def predicted_decode_tok_s(
    *,
    active_params: float,
    bits_per_param: float,
    gpu: MemoryTier,
    ram: MemoryTier,
    disk: MemoryTier,
    kv_bytes: float = 0.0,
) -> dict:
    """Decode throughput ceiling for a memory-tier layout.

    Decode is bandwidth-bound: each token must read every active weight once.
    We fill tiers in preference order (GPU -> RAM -> disk) and take the slowest
    tier as the limiting factor, because tiers are traversed serially within a
    token. Overlap can improve on this; it cannot beat the slowest tier.
    """
    need = bytes_for_params(active_params, bits_per_param)
    remaining = need
    plan = {}
    for tier in (gpu, ram, disk):
        cap = max(0.0, tier.capacity_bytes - (kv_bytes if tier is gpu else 0.0))
        take = min(remaining, cap)
        if take > 0:
            plan[tier.name] = {"bytes": take, "seconds": tier.seconds_for(take)}
        remaining -= take
        if remaining <= 0:
            break
    total_s = sum(v["seconds"] for v in plan.values())
    return {
        "active_bytes": need,
        "placement": plan,
        "unplaceable_bytes": max(0.0, remaining),
        "seconds_per_token": total_s,
        "tok_s": (1.0 / total_s) if total_s > 0 else float("inf"),
        "bottleneck": max(plan, key=lambda k: plan[k]["seconds"]) if plan else None,
    }
