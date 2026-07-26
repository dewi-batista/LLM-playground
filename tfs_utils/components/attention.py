import torch.nn as nn
import torch.nn.functional as F


def build_mha_params(d_model: int, num_heads: int, **kwargs) -> dict:
    return {
        "W_QKV": nn.Linear(d_model, 3 * d_model),
        "W_O": nn.Linear(d_model, d_model),
    }


def mha_forward(block, H):
    B, T, _ = H.shape
    Q, K, V = block.W_QKV(H).chunk(3, dim=-1)
    Q = Q.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)
    K = K.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)
    V = V.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)

    O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    O = O.transpose(1, 2).reshape(B, T, block.num_heads * block.d_head)
    return block.W_O(O)


ATTENTIONS = {
    "mha": {
        "build_params": build_mha_params,
        "forward": mha_forward,
        "resid_param": "W_O",
    },
}
