import warnings

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from vish.tp_model import TPModelFactory
from vish.constants import LEARNING_RATE
from vish.lightning.data.imagenet import ImageNet1kMultiLabelDataModule
from vish.lightning.loss import BELMode
from vish.lightning.modulev2 import BroadFineModelLM, VALIDATION_METRIC_NAME
import torch

logging.set_verbosity_warning()

warnings.filterwarnings("ignore")
seed_everything(42)

# Data Module
datamodule = ImageNet1kMultiLabelDataModule(
    is_test=False,
    depth=7,
)

datamodule.prepare_data()
datamodule.setup()


LOAD_CKPT = False

CKPT_PATH = "/home/students/sandip/from_Ankita/vit_code/logs/imagenet1k/full/depth9/lightning_logs/version_0/checkpoints/tpdualvit-imagenet1k-epoch=15-val_af=0.784.ckpt"

l_module = BroadFineModelLM(
    model=TPModelFactory.get_model("IMAGENET1K"),
    num_fine_outputs=1000,
    num_broad_outputs=datamodule.num_broad_classes,
    lr=LEARNING_RATE,
    loss_mode=BELMode.CLUSTER,
)

checkpoint_callback = ModelCheckpoint(
    monitor=VALIDATION_METRIC_NAME,
    filename="tpdualvit-imagenet1k" + "-{epoch:02d}-" + f"-{{{VALIDATION_METRIC_NAME}:.3f}}",
    save_top_k=2,
    mode="max",
)


kwargs = {
    "max_epochs": 300,
    "accelerator": "gpu",
    "devices": 1,
    "logger": CSVLogger(save_dir="logs/imagenet1k/full/depth7"),
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
    ],
    "num_sanity_val_steps": 2,
    "gradient_clip_val": 1,
    # "precision": 16,
    "accumulate_grad_batches": 2,
}

if LOAD_CKPT:
    kwargs = {**kwargs, "resume_from_checkpoint": CKPT_PATH}

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    if LOAD_CKPT:
        # trainer.test(l_module, datamodule=datamodule, ckpt_path=CKPT_PATH)
        l_module.load_state_dict(torch.load(CKPT_PATH)["state_dict"], strict=False)
        trainer.test(l_module, datamodule=datamodule)
        #pass

    trainer.fit(l_module, datamodule=datamodule)
    #trainer.test(l_module, datamodule=datamodule)
