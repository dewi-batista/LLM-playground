import torch.nn as nn


def build_gelu_mlp_params(d_model: int, d_ff: int, **kwargs) -> dict:
    return {
        "W_1": nn.Linear(d_model, d_ff),
        "W_2": nn.Linear(d_ff, d_model),
        "act": nn.GELU(),
    }


def gelu_mlp_forward(block, H):
    return block.W_2(block.act(block.W_1(H)))


MLPS = {
    "gelu_mlp": {
        "build_params": build_gelu_mlp_params,
        "forward": gelu_mlp_forward,
        "resid_param": "W_2",
    },
}
