import einops
import torch
from torch import nn
from torch.nn import functional as F

from vish.model.common.vit_blocks import (
    MultiHeadAttention,
    TransformerEncoder,
    TransformerBlock,
)


class TPMHA(MultiHeadAttention):
    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 12,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__(emb_dim, num_heads, p_dropout, qkv_bias)
        self.repr_layer = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

    def _split_for_multi_head(self, *args):
        # Split the last dimension to d_h such that d_h * n_heads = emb_dim
        return [
            einops.rearrange(mat, "b s (nh dh) -> b nh s dh", nh=self.num_heads)
            for mat in args
        ]

    def in_proj(self, x, x_ext=None):
        return (
            self.query_layer(x),
            self.key_layer(x),
            self.value_layer(x),
            self.repr_layer(x if x_ext is None else x_ext),
        )

    def forward(self, x, x_ext=None, mask=None):
        queries_mh, keys_mh, values_mh, repr_mh = self._split_for_multi_head(
            *self.in_proj(x, x_ext)
        )

        attn_mat_mh = (queries_mh @ keys_mh.transpose(-2, -1)) / self.norm_factor

        if mask is not None:
            attn_mat_mh = self._add_mask(attn_mat_mh, mask)

        attn_mat_mh = self.dropout(F.softmax(attn_mat_mh, dim=-1))

        # b nh s dh for both fillers and repr_mh
        fillers_mh = attn_mat_mh @ values_mh

        # TPR operation
        return self.tpr_out(repr_mh, fillers_mh)

    def tpr_out(self, repr_mh, fillers_mh):
        # Dimension expected (batch, num_heads, seq_len, d_head)
        # Hadamard product
        tp = torch.multiply(repr_mh, fillers_mh) + fillers_mh
        tp = einops.rearrange(tp, "b nh s dh -> b s (nh dh)")
        #tp = einops.rearrange(fillers_mh, "b nh s dh -> b s (nh dh)")
        return self.out_proj(tp)


class TPTransformerBlock(TransformerBlock):
    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 12,
        pos_wise_ff_dim: int = 3072,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(
            emb_dim,
            num_heads,
            pos_wise_ff_dim,
            p_dropout,
            qkv_bias,
            pwff_bias,
            eps,
        )
        self.mha = TPMHA(emb_dim, num_heads, p_dropout, qkv_bias)

    def _get_mha_residue(self, ip, ip_ext=None, mask=None):
        residue = self.layer_norm_1(ip)
        residue = self.mha(residue, ip_ext, mask)
        residue = self.dropout(residue)
        return residue

    def _get_pwff_residue(self, ip):
        residue = self.layer_norm_2(ip)
        residue = self.pos_wise_ff_layer(residue)
        residue = self.dropout(residue)
        return residue

    def forward(self, x, x_ext=None, mask=None):
        output = x
        # += is inplace operation, may cause errors
        output = output + self._get_mha_residue(output, x_ext, mask)
        output = output + self._get_pwff_residue(output)
        return output


class TPTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        num_layers: int = 12,
        emb_dim: int = 768,
        num_heads: int = 12,
        pwff_dim: int = 3072,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.tf_blocks = nn.ModuleList(
            [
                TPTransformerBlock(
                    emb_dim, num_heads, pwff_dim, p_dropout, qkv_bias, pwff_bias
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, x_ext_list=None, mask=None):
        """
        Forward method of TP Encoder. If we do not want external, send None
        Args:
            x: Input tensor of shape (b, s, d)
            x_ext_list: List[Tensor], num_layers tensor, each of shape (b, s, d)
            mask: MIM

        Returns:
            Tensor

        """
        for idx, (block, x_ext) in enumerate(zip(self.tf_blocks, x_ext_list)):
            x = block(x, x_ext, mask)
        return x
