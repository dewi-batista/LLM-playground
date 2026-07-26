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


def resolve_arch_config(source: dict, overrides: dict | None = None) -> ArchConfig:
    """`source` may be a config.yaml `transformer-{size}` section or a loaded
    checkpoint dict; `overrides` (if given) take precedence over `source` --
    used by SFT to pull seq_len/dropout from the sft-{size} config while the
    rest of the architecture stays frozen to the base checkpoint."""
    src = {**source, **(overrides or {})}
    d_model = int(src["d_model"])
    return ArchConfig(
        d_model=d_model,
        num_blocks=int(src["num_blocks"]),
        dropout=float(src["dropout"]),
        seq_len=int(src["seq_len"]),
        num_heads=int(src["num_heads"]) if "num_heads" in src else d_model // 64,
        d_ff=int(src["d_ff"]) if "d_ff" in src else 4 * d_model,
        pos_encoding=str(src.get("pos_encoding", "sinusoidal")),
        norm=str(src.get("norm", "layernorm")),
        attn=str(src.get("attn", "mha")),
        mlp=str(src.get("mlp", "gelu_mlp")),
    )


@dataclass
class ModelBundle:
    E: nn.Embedding
    model: nn.Sequential
    final_lay_norm: nn.Module
    U: nn.Linear
    pe: torch.Tensor
    arch: ArchConfig


def build_block(arch: ArchConfig) -> TransformerBlock:
    return TransformerBlock(
        arch.d_model,
        arch.d_ff,
        arch.num_heads,
        arch.dropout,
        arch.seq_len,
        norm=arch.norm,
        attn=arch.attn,
        mlp=arch.mlp,
    )


def build_model(arch: ArchConfig, V: int, device: torch.device) -> ModelBundle:
    E = nn.Embedding(V, arch.d_model).to(device)
    final_lay_norm = NORMS[arch.norm](arch.d_model).to(device)
    model = nn.Sequential(*[build_block(arch) for _ in range(arch.num_blocks)]).to(device)
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
        getattr(block, block.attn_resid_param).weight.data /= resid_scale
        getattr(block, block.mlp_resid_param).weight.data /= resid_scale


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
    }
