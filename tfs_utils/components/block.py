import torch.nn as nn

from tfs_utils.components.attention import ATTENTIONS
from tfs_utils.components.mlp import MLPS
from tfs_utils.components.norm import NORMS


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        seq_len: int,
        norm: str = "layernorm",
        attn: str = "mha",
        mlp: str = "gelu_mlp",
        num_kv_heads: int | None = None,
        window_size: int | None = None,
        layer_idx: int | None = None,
        num_experts: int = 1,
        top_k: int = 1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.layer_idx = layer_idx
        self.top_k = top_k
        self._last_aux_loss = None  # set by an MoE mlp_forward; stays None otherwise

        attn_spec = ATTENTIONS[attn]
        attn_params = attn_spec["build_params"](
            d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
        )
        mlp_spec = MLPS[mlp]
        mlp_params = mlp_spec["build_params"](
            d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
        )

        # Assignment order below matches the pre-refactor TransformerBlock exactly
        # (attn params, mlp params, norms, dropouts, "act" last) so that
        # model.parameters() iteration order -- and therefore optimizer_state_dict
        # resume, which torch matches positionally -- stays compatible with
        # checkpoints saved before this refactor. register_buffer calls don't
        # affect this order since buffers aren't parameters.
        for name, module in attn_params.items():
            setattr(self, name, module)
        if "build_buffers" in attn_spec:
            bufs = attn_spec["build_buffers"](
                d_model=d_model, num_heads=num_heads, seq_len=seq_len,
                num_kv_heads=num_kv_heads, window_size=window_size, layer_idx=layer_idx,
            )
            for name, buf in bufs.items():
                self.register_buffer(name, buf, persistent=False)
        for name, module in mlp_params.items():
            if name != "act":
                setattr(self, name, module)

        self.ln1 = NORMS[norm](d_model)
        self.ln2 = NORMS[norm](d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if "act" in mlp_params:
            self.act = mlp_params["act"]

        self.attn_kind = attn
        self.attn_resid_param = attn_spec["resid_param"]
        self.mlp_kind = mlp
        self.mlp_resid_param = mlp_spec["resid_param"]

    def forward(self, X):
        H = self.ln1(X)
        X = X + self.dropout_attn(ATTENTIONS[self.attn_kind]["forward"](self, H))

        H = self.ln2(X)
        X = X + self.dropout_ffn(MLPS[self.mlp_kind]["forward"](self, H))
        return X
