import torch

from vish.model.tp.blocks import TPTransformerEncoder
from vish.model.vanilla.vit import ViTBasicForImageClassification


class TPVitImageClassification(ViTBasicForImageClassification):
    """
    ViT with TPR
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
        mlp_outputs_list: tuple = (10,),
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
        clf_head_bias: bool = False,
        conv_bias: bool = False,
    ):
        super().__init__(
            img_height,
            img_width,
            img_in_channels,
            patch_dim,
            emb_dim,
            num_layers,
            num_attention_heads,
            pwff_hidden_dim,
            num_classification_heads,
            mlp_outputs_list,
            p_dropout,
            qkv_bias,
            pwff_bias,
            clf_head_bias,
            conv_bias,
        )
        self.transformer_encoder = TPTransformerEncoder(
            num_layers,
            emb_dim,
            num_attention_heads,
            pwff_hidden_dim,
            p_dropout,
            qkv_bias,
            pwff_bias,
        )

    def forward(self, x: torch.FloatTensor, x_ext_list: list[torch.FloatTensor] = None):
        if x_ext_list is None:
            x_ext_list = [None for _ in range(self.transformer_encoder.num_layers)]
        op_seq: torch.FloatTensor = x
        op_seq = self.embedding_layer(op_seq)
        op_seq = self._append_additional_tokens(op_seq)
        op_seq = self.positional_embedding(op_seq)
        op_seq = self.transformer_encoder(op_seq, x_ext_list)
        op_seq = self.final_layer_norm(op_seq)

        if self.num_extra_tokens < 1:
            # No extra tokens were added
            print(op_seq.shape)
            return op_seq.mean(dim=(1, 2))

        # Broad to fine
        additional_embeddings = op_seq[:, -self.num_extra_tokens :, :]
        logits = self.to_logits(additional_embeddings)
        return additional_embeddings, logits

    def just_classify(self, x):
        return self.to_logits(x)

    def to_logits(self, additional_embeddings):
        logits = [
            self.mlp_heads[idx](additional_embeddings[:, idx])
            for idx in range(self.num_classification_heads)
        ]
        return logits
