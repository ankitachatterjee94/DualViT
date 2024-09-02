from typing import Literal
from transformers import ViTModel

from vish.constants import IMG_SIZE
from vish.model.tp.modified import TPDualModifiedVit
from vish.model.tp.tp_vit import TPVitImageClassification

# Change as per need, See HuggingFace for available weights
PRETRAINED_MODEL_STRING = f"google/vit-base-patch16-{IMG_SIZE}"

cifar10_model_params = {
    "img_height": IMG_SIZE,
    "img_width": IMG_SIZE,
    "img_in_channels": 3,
    "patch_dim": 16,
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (10,),
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

cifar100_model_params = {
    "img_height": IMG_SIZE,
    "img_width": IMG_SIZE,
    "img_in_channels": 3,
    "patch_dim": 16,
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (100,),
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

imagenet1k_model_params = {
    "img_height": IMG_SIZE,
    "img_width": IMG_SIZE,
    "img_in_channels": 3,
    "patch_dim": 16,
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (1000,),
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}


# This is just a utility class
class TPModelFactory:
    @staticmethod
    def get_model(
        model_name: Literal["CIFAR10", "CIFAR100", "IMAGENET1K"]
    ) -> TPDualModifiedVit:
        broad_model = ViTModel.from_pretrained(PRETRAINED_MODEL_STRING)
        fine_model = None

        match model_name:
            case "CIFAR10":
                fine_model = TPVitImageClassification(**cifar10_model_params)

            case "CIFAR100":
                fine_model = TPVitImageClassification(**cifar100_model_params)

            case "IMAGENET1K":
                fine_model = TPVitImageClassification(**imagenet1k_model_params)

        return TPDualModifiedVit(
            fine_model=fine_model,
            broad_model=broad_model,
            debug=False,
        )
