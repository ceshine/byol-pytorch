import os
from pathlib import Path

import typer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
from byol_pytorch import BYOL, RandomApply
import kornia

from bit_models import KNOWN_MODELS as BIT_MODELS


def load_and_pad_image(filepath: Path) -> Image.Image:
    image = Image.open(filepath).convert('RGB').resize(
        (300, 169), Image.ANTIALIAS)
    # pad the image
    new_image = Image.new("RGB", (300, 256))
    new_image.paste(image, ((0, 43)))
    return np.array(new_image)


# constants
LR = 3e-4
NUM_GPUS = 2
IMAGE_SIZE = 128
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 4

# pytorch lightning module


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, dataset: Dataset, batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=NUM_WORKERS, shuffle=True, pin_memory=True
        )

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.image_size = image_size
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = load_and_pad_image(path)
        return kornia.image_to_tensor(img).float()

# main


def main(arch: str, image_folder: str, from_scratch: bool = False, num_gpus: int = 1, epochs: int = 100):
    if arch.startswith("BiT"):
        base_model = BIT_MODELS[arch](head_size=-1)
        if not from_scratch:
            base_model.load_from(np.load(f"cache/pretrained/{arch}.npz"))
        net_final_size = base_model.width_factor * 2048
    else:
        raise ValueError(f"arch '{arch}'' not supported")
    ds = ImagesDataset(image_folder, IMAGE_SIZE)
    # train_loader = DataLoader(
    #     ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        base_model,
        ds,
        batch_size=4,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        net_final_size=net_final_size,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        # Somehow kornia doesn't work with native nor apex amp here
        # apex amp O2 works with kornia when using pytorch_help_bot, though
        # amp_backend='apex',
        # amp_level='O2', precision=16,
        gpus=num_gpus,
        max_epochs=epochs,
        accumulate_grad_batches=1,
        auto_scale_batch_size='power'
    )

    trainer.tune(model)

    trainer.fit(model)


if __name__ == '__main__':
    typer.run(main)
