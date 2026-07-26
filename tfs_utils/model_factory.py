import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from tfs_utils.components import NORM_INIT_FNS, NORMS, POS_ENCODINGS, TransformerBlock


@dataclass
class ArchConfig:
    d_model: int
    num_heads: int
    num_blocks: int
    d_ff: int
    dropout: float
    seq_len: int
    pos_encoding: str = "sinusoidal"
    norm: str = "layernorm"
    attn: str = "mha"
    mlp: str = "gelu_mlp"
    num_kv_heads: int | None = None  # resolved to num_heads in __post_init__ if left unset
    window_size: int | None = None
    num_experts: int = 1
    top_k: int = 1
    aux_loss_weight: float = 0.01  # only used when mlp="moe_swiglu"

    def __post_init__(self):
        # Keeps the "num_kv_heads is always a concrete int" invariant true
        # regardless of construction path (resolve_arch_config already passes
        # it explicitly, but direct ArchConfig(...) construction, e.g. in
        # tests, should still get the same no-GQA-reduction default).
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


def resolve_arch_config(source: dict, overrides: dict | None = None) -> ArchConfig:
    """`source` may be a config.yaml `transformer-{size}` section or a loaded
    checkpoint dict; `overrides` (if given) take precedence over `source` --
    used by SFT to pull seq_len/dropout from the sft-{size} config while the
    rest of the architecture stays frozen to the base checkpoint."""
    src = {**source, **(overrides or {})}
    d_model = int(src["d_model"])
    num_heads = int(src["num_heads"]) if "num_heads" in src else d_model // 64
    return ArchConfig(
        d_model=d_model,
        num_blocks=int(src["num_blocks"]),
        dropout=float(src["dropout"]),
        seq_len=int(src["seq_len"]),
        num_heads=num_heads,
        d_ff=int(src["d_ff"]) if "d_ff" in src else 4 * d_model,
        pos_encoding=str(src.get("pos_encoding", "sinusoidal")),
        norm=str(src.get("norm", "layernorm")),
        attn=str(src.get("attn", "mha")),
        mlp=str(src.get("mlp", "gelu_mlp")),
        num_kv_heads=int(src["num_kv_heads"]) if "num_kv_heads" in src else num_heads,
        window_size=int(src["window_size"]) if src.get("window_size") is not None else None,
        num_experts=int(src["num_experts"]) if "num_experts" in src else 1,
        top_k=int(src["top_k"]) if "top_k" in src else 1,
        aux_loss_weight=float(src["aux_loss_weight"]) if "aux_loss_weight" in src else 0.01,
    )


@dataclass
class ModelBundle:
    E: nn.Embedding
    model: nn.Sequential
    final_lay_norm: nn.Module
    U: nn.Linear
    pe: torch.Tensor
    arch: ArchConfig


def build_block(arch: ArchConfig, layer_idx: int) -> TransformerBlock:
    return TransformerBlock(
        arch.d_model,
        arch.d_ff,
        arch.num_heads,
        arch.dropout,
        arch.seq_len,
        norm=arch.norm,
        attn=arch.attn,
        mlp=arch.mlp,
        num_kv_heads=arch.num_kv_heads,
        window_size=arch.window_size,
        layer_idx=layer_idx,
        num_experts=arch.num_experts,
        top_k=arch.top_k,
    )


def build_model(arch: ArchConfig, V: int, device: torch.device) -> ModelBundle:
    if arch.attn in ("mha_rope", "gqa_rope") and arch.pos_encoding != "none":
        raise ValueError(
            f"attn={arch.attn!r} requires pos_encoding='none' (got {arch.pos_encoding!r}); "
            "otherwise position would be encoded twice."
        )
    if arch.attn != "gqa_rope" and (arch.window_size is not None or arch.num_kv_heads != arch.num_heads):
        raise ValueError(
            f"window_size/num_kv_heads only apply to attn='gqa_rope' (got attn={arch.attn!r})"
        )
    if arch.mlp != "moe_swiglu" and (arch.num_experts != 1 or arch.top_k != 1):
        raise ValueError(f"num_experts/top_k only apply to mlp='moe_swiglu' (got mlp={arch.mlp!r})")
    if arch.mlp == "moe_swiglu" and not (1 <= arch.top_k <= arch.num_experts):
        raise ValueError(
            f"top_k must be between 1 and num_experts (got top_k={arch.top_k}, num_experts={arch.num_experts})"
        )

    E = nn.Embedding(V, arch.d_model).to(device)
    final_lay_norm = NORMS[arch.norm](arch.d_model).to(device)
    model = nn.Sequential(*[build_block(arch, i) for i in range(arch.num_blocks)]).to(device)
    U = nn.Linear(arch.d_model, V, bias=False).to(device)
    U.weight = E.weight  # weight tying

    pe = POS_ENCODINGS[arch.pos_encoding](arch.seq_len, arch.d_model, device=device)
    return ModelBundle(E=E, model=model, final_lay_norm=final_lay_norm, U=U, pe=pe, arch=arch)


def init_weights(bundle: ModelBundle) -> None:
    """GPT-2-style fresh init (suggested to me) for stability: W elements ~
    N(0, 0.02), biases = 0, norm scale = 1, shift = 0, plus residual-projection
    rescaling by 1/sqrt(2*num_blocks). Only called for a fresh (non-resumed) run."""

    def _init_linear_embedding(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    bundle.E.apply(_init_linear_embedding)
    bundle.model.apply(_init_linear_embedding)
    bundle.final_lay_norm.apply(_init_linear_embedding)

    norm_init_fn = NORM_INIT_FNS[bundle.arch.norm]
    norm_init_fn(bundle.final_lay_norm)
    for block in bundle.model:
        norm_init_fn(block.ln1)
        norm_init_fn(block.ln2)

    resid_scale = math.sqrt(2 * bundle.arch.num_blocks)
    for block in bundle.model:
        _resid_tensor(block, block.attn_resid_param).data /= resid_scale
        _resid_tensor(block, block.mlp_resid_param).data /= resid_scale


def _resid_tensor(block: nn.Module, name: str) -> torch.Tensor:
    # resid_param may name an nn.Module with a .weight (e.g. mha's W_O, a
    # plain nn.Linear) or a raw batched nn.Parameter set directly on the block
    # (e.g. MoE's W_down, which has no wrapping Module) -- handle both.
    target = getattr(block, name)
    return target.weight if isinstance(target, nn.Module) else target


def collect_aux_loss(model: nn.Sequential) -> torch.Tensor:
    """Sums each MoE block's load-balancing loss (stashed as `_last_aux_loss`
    during forward, since TransformerBlock.forward must stay single-tensor-out
    for nn.Sequential/checkpoint() compatibility) into one scalar the training
    loop can add to the main loss. Returns an inert 0.0 for an all-dense model."""
    aux_losses = [block._last_aux_loss for block in model if getattr(block, "_last_aux_loss", None) is not None]
    if not aux_losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(aux_losses).mean()  # mean across MoE layers, not sum, so the weight doesn't scale with depth


def arch_to_ckpt_fields(arch: ArchConfig) -> dict:
    return {
        "d_model": arch.d_model,
        "num_heads": arch.num_heads,
        "num_blocks": arch.num_blocks,
        "d_ff": arch.d_ff,
        "pos_encoding": arch.pos_encoding,
        "norm": arch.norm,
        "attn": arch.attn,
        "mlp": arch.mlp,
        "num_kv_heads": arch.num_kv_heads,
        "window_size": arch.window_size,
        "num_experts": arch.num_experts,
        "top_k": arch.top_k,
        "aux_loss_weight": arch.aux_loss_weight,
    }
