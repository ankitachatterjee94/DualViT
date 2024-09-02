# Decomposed model for CIFAR 10
# TODO: This is hardcoded. Parse hierarchical tree and create blocks regarding the same
# TODO: Need to add positional embedding and additional class token logic
# TODO: Do we at all need [cls] token or can we do mean at the last step of MLP head -no use global pooling
# TODO: How to decide output for block? - By number of classes
# TODO: Add if-else logic for empty batch


from pprint import pprint
from typing import Dict, Tuple

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from vish.model.decomposed.split import segregate_samples_within_batch
from vish.model.common.lang import PreTrainedWordEmbeddings
from vish.model.common.tree import LabelHierarchyTree
from vish.model.common.vit_blocks import TransformerBlockGroup

LANG_MODEL_NAME = "distilbert-base-uncased"
CIFAR_10_HIERARCHICAL_LABEL_PATH = "vish/data/cifar10.xml"


lang_model = PreTrainedWordEmbeddings(LANG_MODEL_NAME)
label_tree = LabelHierarchyTree(CIFAR_10_HIERARCHICAL_LABEL_PATH)

net_l1 = TransformerBlockGroup(num_blocks=4)
net_l2 = {
    "animal": TransformerBlockGroup(num_blocks=4),
    "vehicle": TransformerBlockGroup(num_blocks=4),
}
net_l3 = {
    "heavy": TransformerBlockGroup(num_blocks=4),
    "light": TransformerBlockGroup(num_blocks=4),
    "domestic": TransformerBlockGroup(num_blocks=4),
    "herbivore": TransformerBlockGroup(num_blocks=4),
    "other_animal": TransformerBlockGroup(num_blocks=4),
}


@DeprecationWarning
class TransformerDecomposed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net_l1 = net_l1
        self.net_l2 = net_l2
        self.net_l3 = net_l3
        self.segregator = segregate_samples_within_batch
        self.num_classes = 10  # change

    def get_labels_and_pred(
        self,
        pred_labels_dict: Dict,
        fine_labels: int,
    ) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Return output from model and expanded labels
        Order the labels_dict and expand the fine_label

        y_pred, y_true
        """
        one_hot_fine_labels = F.one_hot(fine_labels, self.num_classes)
        preds = torch.cat(tuple(pred_labels_dict.values()), dim=-1)  # [B, C]
        return preds, one_hot_fine_labels

    def global_pool_sequence(self, batch):
        return torch.mean(batch, dim=-2)  # [B, D]

    def reshape_external_queries(self, q, batch_dim):
        return repeat(q, "c d -> b c d", b=batch_dim)

    def segregate_batch(self, x, words):
        """
        Takes a batch and segregates into samples belonging to different super classes

        words: word vectors at same level. Eg for CIFAR 10, we can have this as
        ('animal', 'vehicle')
        """
        batch_dim, *_ = x.shape
        external_queries = torch.stack([lang_model(word) for word in words])
        external_queries = self.reshape_external_queries(external_queries, batch_dim)
        _, output_dict = self.segregator(external_queries, x, x)
        return {word: output_dict[idx] for idx, word in enumerate(words)}

    def print_tensor_dict(self, t_dict):
        # Useful for debugging dicitonaries
        d = {key: value.shape for key, value in t_dict.items()}
        pprint(f"tensor dict is: {d}")

    def forward(self, batch):
        # Initial Block
        output = self.net_l1(batch)

        # Level 1 split
        super_class_label_1 = [item[0] for item in label_tree.get_elements_at_depth(1)]
        seg_samples_1 = self.segregate_batch(output, super_class_label_1)

        output_1 = {
            label: self.net_l2[label](seg_samples_1[label])
            for label in super_class_label_1
        }

        # Level 2 split
        super_class_label_2_animal = [
            item[0] for item in label_tree.get_immediate_children("animal")
        ]
        super_class_label_2_vehicle = [
            item[0] for item in label_tree.get_immediate_children("vehicle")
        ]

        super_class_label_2_total = [
            item[0] for item in label_tree.get_elements_at_depth(2)
        ]

        animal_sub_output = self.segregate_batch(
            output_1["animal"], super_class_label_2_animal
        )
        vehicle_sub_output = self.segregate_batch(
            output_1["vehicle"], super_class_label_2_vehicle
        )

        seg_samples_2 = {**animal_sub_output, **vehicle_sub_output}
        output_2 = {
            label: self.net_l3[label](seg_samples_2[label])
            for label in super_class_label_2_total
        }

        # Level 3
