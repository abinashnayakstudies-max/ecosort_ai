"""
dataset.py - PyTorch Dataset + DataLoaders for data/{train,val,test}/{class}/*.jpg

A Dataset only needs two methods: __len__ and __getitem__(i). The DataLoader
then batches and shuffles them for us.
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import (CLASS_NAMES, CLASS_TO_IDX, DATA_DIR, IMG_EXTS,
                        DEFAULT_IMG_SIZE, SEED)
from src.preprocessing import get_train_transform, get_eval_transform


class WasteDataset(Dataset):
    def __init__(self, split_dir, transform=None, max_per_class=None, seed=SEED):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples = []                       # list of (path, label_index)
        rng = random.Random(seed)
        for cls in CLASS_NAMES:                 # fixed class order -> fixed labels
            files = sorted(p for p in (self.split_dir / cls).glob("*")
                           if p.suffix.lower() in IMG_EXTS)
            if max_per_class:                   # only used for quick smoke tests
                rng.shuffle(files)
                files = files[:max_per_class]
            self.samples += [(p, CLASS_TO_IDX[cls]) for p in files]
        if not self.samples:
            raise FileNotFoundError(f"No images in {self.split_dir}. Run prepare_dataset.py first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = [0] * len(CLASS_NAMES)
        for _, l in self.samples:
            counts[l] += 1
        return counts


def get_class_weights(counts):
    """Rarer class -> bigger weight in the loss, so the model can't ignore it."""
    total, k = sum(counts), len(counts)
    return torch.tensor([total / (k * max(c, 1)) for c in counts], dtype=torch.float)


def make_loader(split, img_size=DEFAULT_IMG_SIZE, batch_size=32, augment=False,
                num_workers=2, max_per_class=None, data_dir=DATA_DIR, seed=SEED):
    tf = get_train_transform(img_size, augment) if split == "train" else get_eval_transform(img_size)
    ds = WasteDataset(Path(data_dir) / split, tf, max_per_class, seed)
    g = torch.Generator().manual_seed(seed)       # reproducible shuffling
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=num_workers, generator=g,
                        pin_memory=torch.cuda.is_available())
    return ds, loader
