import numpy as np
import hashlib
from torchvision import datasets
from PIL import Image
from torch.utils.data.dataloader import default_collate
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class SimpleVisionExample:
    path: Path
    label: Optional[int]
    latent: Optional[torch.Tensor]

    def hash_str(self, salt):
        return int(hashlib.md5(f"{salt}{str(self.path)}".encode("utf-8")).hexdigest(), 16)

    def to_dict(self):
        return {
            "path": str(self.path),
            "label": self.label,
            "latent": self.latent.numpy().tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            path=Path(d["path"]),
            label=d["label"],
            latent=torch.tensor(np.array(d["latent"], dtype=np.float), dtype=torch.float),
        )


def _split_range(splits, split_idx):
    sum_splits = np.cumsum(splits, 0)

    if sum_splits[-1] != 1.0:
        raise RuntimeError(f"Splits must sum to 1 (actual: {sum_splits[-1]})")
    elif split_idx >= len(sum_splits):
        raise RuntimeError(
            f"Invalid split index {split_idx} (must be less than {len(sum_splits)})"
        )

    if split_idx == 0:
        start_range = 0.0
    else:
        start_range = sum_splits[split_idx - 1]

    end_range = sum_splits[split_idx]

    return (start_range, end_range)


def _in_split_range(split_range, item, salt):
    start_range, end_range = split_range
    val = item.hash_str(salt) % 100000 / 100000
    return (val >= start_range and val < end_range).item()


def split_train_valid_test(
    examples: List[SimpleVisionExample],
    train_size=0.8,
    valid_size=0.1,
    test_size=0.1,
    split_salt="thesepretzelsaretoosalty",
):
    dataset_train_valid_test_split = [train_size, valid_size, test_size]
    train_range, valid_range, test_range = [
        _split_range(dataset_train_valid_test_split, i)
        for i in range(len(dataset_train_valid_test_split))
    ]
    train_set, valid_set, test_set = [
        [e for e in examples if _in_split_range(r, e, split_salt)]
        for r in (train_range, valid_range, test_range)
    ]
    print(
        f"Dataset sizes train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}"
    )
    return train_set, valid_set, test_set


def pil_loader(path, mode="RGB"):
    with open(path, "rb") as f:
        img = Image.open(f)
        if mode != "RGBA" and img.mode == "RGBA":
            image = Image.new("RGBA", img.size, "WHITE")
            image.paste(img, (0, 0), img) 
            return image.convert(mode)
        else:
            return img.convert(mode)


class SimpleVisionDataset(datasets.VisionDataset):
    def __init__(self, examples: List[SimpleVisionExample], transform=None, mode="RGB"):
        super().__init__("", transform=transform, target_transform=None)
        self.examples = examples
        self.mode = mode

    def __getitem__(self, index):
        row = self.examples[index]
        img = pil_loader(row.path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, row.latent, row.label

    @property
    def collate_fn(self):
        def collate(batch):
            bl = len(batch[0])
            return tuple(
                default_collate([e[i] for e in batch])
                for i in range(bl)
            )
        return collate

    def __len__(self):
        return len(self.examples)