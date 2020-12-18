import os
import math
from pathlib import Path

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
            self, net, train_dataset: Dataset, valid_dataset: Dataset, epochs: int, learning_rate: float,
            batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.learner = BYOL(net, **kwargs)

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

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.batch_size,
            num_workers=NUM_WORKERS, shuffle=False, pin_memory=True,
            drop_last=False
        )

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # 2. log `val_loss`
        self.log('val_loss', loss)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        n_steps = math.floor(len(self.train_dataset) / self.batch_size) * self.epochs
        lr_durations = [
            int(n_steps*0.05),
            int(np.ceil(n_steps*0.95)) + 1
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            ),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


def main(arch: str, image_folder: str, from_scratch: bool = False, num_gpus: int = 1, epochs: int = 100, lr: float = 4e-4):
    if arch.startswith("BiT"):
        base_model = BIT_MODELS[arch](head_size=-1)
        if not from_scratch:
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
        augment_fn=torch.nn.Sequential(
            T.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p=0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        ),
        batch_size=4,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        net_final_size=net_final_size,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        amp_level='O2', precision=16,
        gpus=num_gpus,
        val_check_interval=1000,
        max_epochs=epochs,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                monitor='val_loss',
                filename='byol-{step:06d}-{val_loss:.2f}',
                save_top_k=2)
        ],
        accumulate_grad_batches=2,
        auto_scale_batch_size='power'
    )

    trainer.tune(model)

    trainer.fit(model)


if __name__ == '__main__':
    typer.run(main)
