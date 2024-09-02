import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTConfig


class VitImageClassificationBroadFine(ViTForImageClassification):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.num_outputs = None

    def pre_forward_adjust(
        self,
        num_output_nodes: tuple,
    ) -> None:
        """Adjust model before forward

        Args:
            num_output_nodes (tuple): The number of labels. Values should be from broad to fine.
             That is last value should be the finest class while first should be the broadest
        """
        embs = self.vit.embeddings
        self.num_outputs = len(num_output_nodes)
        num_patches = embs.patch_embeddings.num_patches
        embs.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + self.num_outputs, self.config.hidden_size)
        )
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, num_output_nodes[i])
                for i in range(len(num_output_nodes))
            ]
        )
        embs.cls_token = nn.Parameter(
            torch.randn(1, len(num_output_nodes), self.config.hidden_size)
        )  # broad,..., fine

    def forward(self, x, *args, **kwargs):
        outputs = self.vit.layernorm(self.vit(x)[0])

        op_list = []
        emb_list = []

        for idx in range(self.num_outputs):
            t = outputs[:, idx]
            op_list.append(self.classifier[idx](t))
            emb_list.append(t)

        return emb_list, op_list
