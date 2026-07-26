import torch
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


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def build_rope_cache(seq_len: int, d_head: int, base: float = 10000.0):
    assert d_head % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def build_mha_rope_buffers(d_model: int, num_heads: int, seq_len: int, **kwargs) -> dict:
    d_head = d_model // num_heads
    cos, sin = build_rope_cache(seq_len, d_head)
    return {"rope_cos": cos, "rope_sin": sin}


def mha_rope_forward(block, H):
    B, T, _ = H.shape
    Q, K, V = block.W_QKV(H).chunk(3, dim=-1)
    Q = Q.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)
    K = K.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)
    V = V.reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)

    # buffers are fp32; Q/K may be bf16 under autocast -- elementwise ops
    # aren't autocast-managed the way matmuls are, so cast explicitly.
    cos = block.rope_cos[:T].to(dtype=Q.dtype)
    sin = block.rope_sin[:T].to(dtype=Q.dtype)
    Q, K = apply_rope(Q, cos, sin), apply_rope(K, cos, sin)

    O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    O = O.transpose(1, 2).reshape(B, T, block.num_heads * block.d_head)
    return block.W_O(O)


def build_gqa_rope_params(d_model: int, num_heads: int, *, num_kv_heads: int | None = None, **kwargs) -> dict:
    num_kv_heads = num_kv_heads or num_heads
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads for GQA"
    d_head = d_model // num_heads
    return {
        "W_Q": nn.Linear(d_model, d_model),
        "W_KV": nn.Linear(d_model, 2 * num_kv_heads * d_head),
        "W_O": nn.Linear(d_model, d_model),
        "attn_sink": nn.Parameter(torch.zeros(num_heads)),
    }


def build_causal_window_mask(seq_len: int, window: int | None) -> torch.Tensor:
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) < window
    return torch.where(allowed, 0.0, float("-inf"))


def build_gqa_rope_buffers(
    d_model: int,
    num_heads: int,
    seq_len: int,
    *,
    window_size: int | None = None,
    layer_idx: int | None = None,
    **kwargs,
) -> dict:
    d_head = d_model // num_heads
    cos, sin = build_rope_cache(seq_len, d_head)
    # gpt-oss alternates full-causal and windowed-local attention every other
    # layer -- a fixed architectural constant, not a config-driven pattern.
    is_windowed = window_size is not None and layer_idx is not None and layer_idx % 2 == 1
    mask = build_causal_window_mask(seq_len, window_size if is_windowed else None)
    return {"rope_cos": cos, "rope_sin": sin, "attn_bias_base": mask}


def gqa_rope_forward(block, H):
    B, T, _ = H.shape
    Q = block.W_Q(H).reshape(B, T, block.num_heads, block.d_head).transpose(1, 2)
    kv_dim = block.W_KV.out_features // 2
    num_kv_heads = kv_dim // block.d_head
    K, V = block.W_KV(H).chunk(2, dim=-1)
    K = K.reshape(B, T, num_kv_heads, block.d_head).transpose(1, 2)
    V = V.reshape(B, T, num_kv_heads, block.d_head).transpose(1, 2)

    cos = block.rope_cos[:T].to(dtype=Q.dtype)
    sin = block.rope_sin[:T].to(dtype=Q.dtype)
    Q, K = apply_rope(Q, cos, sin), apply_rope(K, cos, sin)

    # extend K/V with one dummy all-zero key/value row for the sink column:
    # Q . 0 == 0, so the sink's pre-softmax score is purely the additive bias
    # below, and a zero value row means any softmax mass it absorbs
    # contributes nothing to the output.
    K_ext = F.pad(K, (0, 0, 0, 1))
    V_ext = F.pad(V, (0, 0, 0, 1))

    base_mask = block.attn_bias_base[:T, :T].to(dtype=Q.dtype)  # (T, T), 0/-inf
    sink_col = block.attn_sink.to(dtype=Q.dtype).view(-1, 1, 1).expand(-1, T, 1)  # (H, T, 1)
    attn_bias = torch.cat([base_mask.unsqueeze(0).expand(block.num_heads, -1, -1), sink_col], dim=-1)

    O = F.scaled_dot_product_attention(Q, K_ext, V_ext, attn_mask=attn_bias, enable_gqa=True)
    O = O.transpose(1, 2).reshape(B, T, block.num_heads * block.d_head)
    return block.W_O(O)


ATTENTIONS = {
    "mha": {
        "build_params": build_mha_params,
        "forward": mha_forward,
        "resid_param": "W_O",
    },
    "mha_rope": {
        "build_params": build_mha_params,  # identical shapes -- reused as-is, no GQA yet
        "forward": mha_rope_forward,
        "resid_param": "W_O",
        "build_buffers": build_mha_rope_buffers,
    },
    "gqa_rope": {
        "build_params": build_gqa_rope_params,
        "forward": gqa_rope_forward,
        "resid_param": "W_O",
        "build_buffers": build_gqa_rope_buffers,
    },
}
