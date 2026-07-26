import torch.nn as nn


def build_layernorm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model)


def init_layernorm(m: nn.Module) -> None:
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)


def build_rmsnorm(d_model: int) -> nn.Module:
    return nn.RMSNorm(d_model)


def init_rmsnorm(m: nn.Module) -> None:
    nn.init.ones_(m.weight)


NORMS = {
    "layernorm": build_layernorm,
    "rmsnorm": build_rmsnorm,
}

NORM_INIT_FNS = {
    "layernorm": init_layernorm,
    "rmsnorm": init_rmsnorm,
}
