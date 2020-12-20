import os
import math
from pathlib import Path
from typing import Optional
from itertools import chain

import typer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from byol_pytorch import BYOL, RandomApply
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_helper_bot import LinearLR, MultiStageScheduler
from pytorch_helper_bot.optimizers import RAdam

from bit_models import KNOWN_MODELS as BIT_MODELS
from dataset import get_datasets, IMAGE_SIZE


def load_and_pad_image(filepath: Path) -> Image.Image:
    image = Image.open(filepath).convert('RGB').resize(
        (300, 169), Image.ANTIALIAS)
    # pad the image
    new_image = Image.new("RGB", (300, 256))
    new_image.paste(image, ((0, 43)))
    return new_image


NUM_WORKERS = 4


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(
            self, net, train_dataset: Dataset, valid_dataset: Dataset,
            epochs: int, learning_rate: float,
            batch_size: int = 32, num_gpus: int = 1, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.learner = BYOL(net, **kwargs)
        self.num_gpus = num_gpus

    def forward(self, images):
        return self.learner(images)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=NUM_WORKERS, shuffle=True, pin_memory=True,
            drop_last=True
        )

    def get_progress_bar_dict(self):
        # don't show the experiment version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, images, _):
        loss = self.forward(images)
        # opt = self.optimizers()
        # self.manual_backward(loss, opt)
        # opt.step()
        # opt.zero_grad()
        self.log("loss", loss, sync_dist=True)
        # print(loss)
        return loss

    def configure_optimizers(self):
        layer_groups = [
            [self.learner.online_encoder.net],
            [
                self.learner.online_encoder.projector,
                self.learner.online_predictor
            ]
        ]
        optimizer = torch.optim.Adam([
            {
                "params": chain.from_iterable([x.parameters() for x in layer_groups[0]]),
                "lr": self.learning_rate * 0.2
            },
            {
                "params": chain.from_iterable([x.parameters() for x in layer_groups[1]]),
                "lr": self.learning_rate
            },
            # {
            #     "params": self.learner.online_predictor.parameters(),
            #     "lr": self.learning_rate
            # },
            # {
            #     "params": self.learner.online_encoder.projector.parameters(),
            #     "lr": self.learning_rate
            # }
        ])
        print(optimizer)
        return optimizer

    # def on_before_zero_grad(self, _):
    #     if self.learner.use_momentum:
    #         self.learner.update_moving_average()


def main(
        arch: str, image_folder: str, from_scratch: bool = False,
        batch_size: Optional[int] = None,
        from_model: Optional[str] = None,
        grad_accu: int = 1,
        num_gpus: int = 1, epochs: int = 100, lr: float = 4e-4):
    if arch.startswith("BiT"):
        base_model = BIT_MODELS[arch](head_size=-1)
        if not from_scratch and not from_model:
            print("Loading pretrained model...")
            base_model.load_from(np.load(f"cache/pretrained/{arch}.npz"))
        net_final_size = base_model.width_factor * 2048
    else:
        raise ValueError(f"arch '{arch}'' not supported")
    train_ds, valid_ds = get_datasets(image_folder, val_ratio=0.05)

    model = SelfSupervisedLearner(
        base_model,
        train_ds,
        valid_ds,
        epochs,
        lr,
        num_gpus=num_gpus,
        batch_size=batch_size if batch_size else 4,
        image_size=IMAGE_SIZE,
        projection_size=256,
        projection_hidden_size=4096,
        net_final_size=net_final_size,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        accelerator='ddp' if num_gpus > 1 else None,
        amp_backend="apex", amp_level='O2',
        precision=16,
        gpus=[1],  # num_gpus,
        val_check_interval=0.5,
        # gradient_clip_val=10,
        max_epochs=epochs,
    )

    trainer.fit(model)


if __name__ == '__main__':
    typer.run(main)
