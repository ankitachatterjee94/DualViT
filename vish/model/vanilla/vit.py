from einops.layers.torch import Rearrange
from torch import nn
from vish.model.common.vit_blocks import PositionalEmbedding1D, TransformerEncoder
import torch
import einops


class ViTBasicForImageClassification(nn.Module):
    """
    Basic ViT Model for Image Classification
    NOTE:
    - num_classification_heads = num of extra tokens added.
    - For [class] token, set it to 1
    - As classification output is expected, if set to < 1, the entire output sequence will be averaged
    - `mlp_outputs_list` takes outputs with broad classes first then fine labels
    - Fine labels is always last output
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        img_in_channels: int = 3,
        patch_dim: int = 16,
        emb_dim: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        pwff_hidden_dim: int = 3072,
        num_classification_heads: int = 1,
        mlp_outputs_list: tuple = (10,),  # Cifar 10 default
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
        clf_head_bias: bool = False,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_dim = patch_dim
        self.emb_dim = emb_dim
        self.num_extra_tokens = self.num_classification_heads = num_classification_heads
        self.num_patch_width = img_width // patch_dim
        self.num_patch_height = img_height // patch_dim

        self.seq_len = (
            self.num_patch_width * self.num_patch_height
        ) + self.num_extra_tokens

        self.embedding_layer = nn.Sequential(
            nn.Conv2d(
                img_in_channels,
                emb_dim,
                kernel_size=(patch_dim, patch_dim),
                stride=(patch_dim, patch_dim),
                bias=conv_bias,
            ),
            Rearrange("b d ph pw -> b (ph pw) d"),
        )

        self.additional_class_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, self.emb_dim))
                for _ in range(self.num_extra_tokens)
            ]
        )

        self.mlp_heads = nn.ModuleList(
            [
                nn.Linear(emb_dim, mlp_outputs_list[idx], bias=clf_head_bias)
                for idx in range(self.num_classification_heads)
            ]
        )

        # 1D positional embedding
        self.positional_embedding = PositionalEmbedding1D(self.seq_len, self.emb_dim)
        self.transformer_encoder = TransformerEncoder(
            num_layers,
            emb_dim,
            num_attention_heads,
            pwff_hidden_dim,
            p_dropout,
            qkv_bias,
            pwff_bias,
        )
        self.final_layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-06)

    def _append_additional_tokens(self, x):
        """
        Appends additional tokens to the input.
        Note: First fine class labels appear
        """
        batch_dim = x.shape[0]
        expanded_tokens = [
            einops.repeat(token, "1 s d -> b s d", b=batch_dim)
            for token in self.additional_class_tokens
        ]
        return torch.cat([x, *expanded_tokens], dim=1)

    def forward(self, x: torch.tensor):
        op_seq: torch.tensor = x
        op_seq = self.embedding_layer(op_seq)
        op_seq = self._append_additional_tokens(op_seq)
        op_seq = self.positional_embedding(op_seq)
        op_seq = self.transformer_encoder(op_seq)
        op_seq = self.final_layer_norm(op_seq)

        if self.num_extra_tokens < 1:
            # No extra tokens were added
            print(op_seq.shape)
            return op_seq.mean(dim=(1, 2))

        # Broad to fine
        op_additional_tokens = op_seq[:, -self.num_extra_tokens :, :]
        return [
            self.mlp_heads[idx](op_additional_tokens[:, idx])
            for idx in range(self.num_classification_heads)
        ]
