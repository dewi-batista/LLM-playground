import torch.nn as nn


def build_layernorm(d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model)


def init_layernorm(m: nn.Module) -> None:
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)


NORMS = {
    "layernorm": build_layernorm,
}

NORM_INIT_FNS = {
    "layernorm": init_layernorm,
}
