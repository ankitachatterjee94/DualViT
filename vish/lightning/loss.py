from enum import Enum

import einops
import torch
from torch import nn


class BELMode(Enum):
    CLUSTER: str = "cluster"
    M3M: str = "m3m"


class BroadFineEmbeddingLoss(nn.Module):
    """
    Broad Fine Embedding Loss
    """

    def __init__(self, num_broad_classes, norm_order: int = 1) -> None:
        super().__init__()
        self.num_broad_classes = num_broad_classes
        self.norm_order = norm_order

    def broad_criterion(self, broad_mean, fine_mean):
        return torch.linalg.norm(broad_mean - fine_mean, self.norm_order)

    @staticmethod
    def adjust_class_token_shape(x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        if len(x.shape) == 3:
            # B 1 D -> B D
            return x.squeeze(1)
        return x

    @staticmethod
    def _mean(x):
        return einops.reduce(x, "b d ->  1 d", "mean")

    def get_m3m_loss(self, broad_mean, fine_this_idx):
        # Mean v/s Mean of Means
        # [1, D] for all index -> [Z, D]
        fine_mean = torch.mean(torch.cat(fine_this_idx, dim=0), 0)
        return self.broad_criterion(broad_mean, fine_mean)

    def get_cluster_loss(self, broad_mean, fine_this_idx):
        # Cluster like loss, mu_center - x_i for all i
        # [1, D] for all index -> [Z, D]
        fine_mean = torch.cat(fine_this_idx, dim=0)
        return self.broad_criterion(broad_mean, fine_mean)

    def forward(
        self,
        broad_class_token,
        broad_labels,
        fine_class_token,
        fine_labels,
        mode: BELMode = BELMode.M3M,
    ) -> torch.Tensor:
        """Method to calculate Embedding Loss for Broad and Fine Labels

        Args:
            broad_class_token (torch.Tensor): Broad Class Embedding Token of shape [B, 1, D] or [B, D]
            broad_labels (torch.Tensor): Broad Class Labels of shape [B, ]
            fine_class_token (torch.Tensor): Fine Class Embedding Token of shape [B, 1, D] or [B, D]
            fine_labels (torch.Tensor): Fine Class Labels of shape [B, ]
            mode (BELMode): Mode of Final loss reduction, Cluster/M3M

        Returns:
            list[(torch.Tensor)]: List of embedding losses for each broad class present
        """
        embedding_losses = []
        # [B, D] dimensional tensor ensured for all class tokens
        broad_class_token = self.adjust_class_token_shape(broad_class_token)
        fine_class_token = self.adjust_class_token_shape(fine_class_token)

        for broad_idx in range(self.num_broad_classes):
            broad_label_filter = broad_labels == broad_idx
            fine_indexes = torch.unique(fine_labels[broad_label_filter])

            if len(fine_indexes) == 0:
                # If no fine labels corresponding to broad label, Empty, then no loss
                continue

            # [1, D]
            broad_mean = self._mean(broad_class_token[broad_label_filter])

            # All Elements of shape [1, D]
            fine_this_idx = [
                self._mean(fine_class_token[(fine_labels == fine_idx)])
                for fine_idx in fine_indexes
            ]

            # Loss as per modes
            if mode == BELMode.M3M:
                idx_loss = self.get_m3m_loss(broad_mean, fine_this_idx)
            elif mode == BELMode.CLUSTER:
                idx_loss = self.get_cluster_loss(broad_mean, fine_this_idx)

            embedding_losses.append(idx_loss)
        return embedding_losses
