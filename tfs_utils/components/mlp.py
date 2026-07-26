import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gelu_mlp_params(d_model: int, d_ff: int, **kwargs) -> dict:
    return {
        "W_1": nn.Linear(d_model, d_ff),
        "W_2": nn.Linear(d_ff, d_model),
        "act": nn.GELU(),
    }


def gelu_mlp_forward(block, H):
    return block.W_2(block.act(block.W_1(H)))


def build_swiglu_params(d_model: int, d_ff: int, **kwargs) -> dict:
    # d_ff is used as-is (not auto-rescaled to the customary ~8/3*d_model SwiGLU
    # convention) so the persisted ArchConfig.d_ff always matches what was built.
    # At equal d_ff this has 3 matrices vs. gelu_mlp's 2 (1.5x the MLP params).
    return {
        "W_gate": nn.Linear(d_model, d_ff, bias=False),
        "W_up": nn.Linear(d_model, d_ff, bias=False),
        "W_down": nn.Linear(d_ff, d_model, bias=False),
    }


def swiglu_forward(block, H):
    return block.W_down(F.silu(block.W_gate(H)) * block.W_up(H))


def build_moe_swiglu_params(d_model: int, d_ff: int, *, num_experts: int = 1, **kwargs) -> dict:
    # Per-expert weights are raw batched Parameters (not num_experts separate
    # nn.Linear submodules) to avoid blowing up per-block attribute count.
    # They aren't visited by init_weights' generic .apply() pass (which only
    # walks nn.Linear/nn.Embedding submodules), so they self-initialize here.
    W_gate = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
    W_up = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
    W_down = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
    for w in (W_gate, W_up, W_down):
        nn.init.normal_(w, mean=0.0, std=0.02)
    return {
        "router": nn.Linear(d_model, num_experts),
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
    }


def moe_swiglu_forward(block, H):
    logits = block.router(H)  # (B, T, E)
    probs = F.softmax(logits, dim=-1)
    topk_vals, topk_idx = probs.topk(block.top_k, dim=-1)
    gate = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize among selected experts
    full_gate = torch.zeros_like(probs)
    full_gate.scatter_(-1, topk_idx, gate)  # (B, T, E), zero for non-selected experts

    # Dense compute: every expert processes every token, then results are
    # masked/combined by gate weight -- simpler than sparse dispatch, and fine
    # at this repo's toy num_experts scale.
    gate_all = torch.einsum("btd,edf->btef", H, block.W_gate)
    up_all = torch.einsum("btd,edf->btef", H, block.W_up)
    hidden = F.silu(gate_all) * up_all
    down_all = torch.einsum("btef,efd->bted", hidden, block.W_down)
    output = torch.einsum("bte,bted->btd", full_gate, down_all)

    # Switch-Transformer-style load-balancing auxiliary loss, stashed as a
    # side-channel attribute (verified safe under non-reentrant grad
    # checkpointing and torch.compile) since TransformerBlock.forward must
    # stay single-tensor-out for nn.Sequential/checkpoint() compatibility.
    f = (full_gate > 0).float().mean(dim=(0, 1))  # fraction of tokens routed to each expert
    P = probs.mean(dim=(0, 1))  # mean router probability per expert
    num_experts = block.W_gate.shape[0]
    block._last_aux_loss = num_experts * (f * P).sum()

    return output


MLPS = {
    "gelu_mlp": {
        "build_params": build_gelu_mlp_params,
        "forward": gelu_mlp_forward,
        "resid_param": "W_2",
    },
    "swiglu": {
        "build_params": build_swiglu_params,
        "forward": swiglu_forward,
        "resid_param": "W_down",
    },
    "moe_swiglu": {
        "build_params": build_moe_swiglu_params,
        "forward": moe_swiglu_forward,
        "resid_param": "W_down",
    },
}
