import glob
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit
from torchvision import transforms


def load_and_pad_image(filepath: Path) -> Image.Image:
    image = Image.open(filepath).convert('RGB').resize(
        (300, 169), Image.ANTIALIAS)
    # pad the image
    new_image = Image.new("RGB", (300, 256))
    new_image.paste(image, ((0, 43)))
    return np.array(new_image)


def collect_images(parse_path, val_ratio: float = 0.2):
    entries = []
    for item in glob.glob(parse_path + "**/*.jpg", recursive=True):
        entries.append(Path(item).resolve())
    df = pd.DataFrame({"path": entries})

    ss = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, valid_idx = next(iter(ss.split(df)))
    return {
        "train": df.iloc[train_idx],
        "valid": df.iloc[valid_idx]
    }


TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])
IMAGE_SIZE = 224


class ByolDataset(Dataset):
    def __init__(self, df: str):
        super().__init__()
        self._df = df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_and_pad_image(item.path)
        image_tensor = TRANSFORM(image)
        return image_tensor


def get_datasets(parse_path: str, val_ratio: float = 0.1):
    data = collect_images(parse_path, val_ratio)
    train_ds = ByolDataset(data["train"])
    valid_ds = ByolDataset(data["valid"])
    print(f"Train size: {len(train_ds)}")
    print(f"Valid size: {len(valid_ds)}")
    return train_ds, valid_ds
