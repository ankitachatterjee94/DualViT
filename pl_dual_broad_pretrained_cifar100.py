import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from vish.tp_model import TPModelFactory
from vish.constants import LEARNING_RATE
from vish.lightning.data.cifar import CIFAR100MultiLabelDataModule
from vish.lightning.data.common import train_transform, test_transform
from vish.lightning.loss import BELMode
from vish.lightning.modulev2 import BroadFineModelLM, VALIDATION_METRIC_NAME
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

# Data Module
datamodule = CIFAR100MultiLabelDataModule(
    is_test=False,
    train_transform=train_transform,
    val_transform=test_transform,
)

datamodule.prepare_data()
datamodule.setup()


LOAD_CKPT = True

CKPT_PATH = "/home/students/sandip/from_Ankita/vit_code/logs/cifar100/modified_dual_tpvit_fulldataset/lightning_logs/version_0/checkpoints/tpdualvitcifar100-epoch=78-val_acc_fine=0.865.ckpt"

checkpoint_callback = ModelCheckpoint(
    monitor=VALIDATION_METRIC_NAME,
    filename="CIFAR100-TpDualViT-p16-384"
    + "-{epoch:02d}-"
    + f"{{{VALIDATION_METRIC_NAME}:.3f}}",
    save_top_k=2,
    mode="max",
)

l_module = BroadFineModelLM(
    model=TPModelFactory.get_model("CIFAR100"),
    num_fine_outputs=100,
    num_broad_outputs=20,
    lr=LEARNING_RATE,
    loss_mode=BELMode.CLUSTER,
)


kwargs = {
    "max_epochs": 300,
    "accelerator": "gpu",
    "gpus": 1,
    "logger": CSVLogger(save_dir="logs/cifar100/modified_dual_tpvit_fulldataset"),
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
    ],
    "num_sanity_val_steps": 5,
    "gradient_clip_val": 1,
}

if LOAD_CKPT:
    kwargs = {**kwargs, "resume_from_checkpoint": CKPT_PATH}

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    if LOAD_CKPT:
        #trainer.test(l_module, datamodule=datamodule, ckpt_path=CKPT_PATH)
        l_module.load_state_dict(torch.load(CKPT_PATH)["state_dict"], strict=False)
        trainer.test(l_module, datamodule=datamodule)

    #trainer.fit(l_module, datamodule=datamodule)
    #trainer.test(l_module, datamodule=datamodule)
