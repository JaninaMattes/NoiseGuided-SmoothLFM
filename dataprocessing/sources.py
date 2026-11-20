"""Source images for dataset generation, read from class folders.

Two folder layouts are supported:

* ``class_<id>/``   — the class index is part of the folder name (layout of the original thesis data);
* ``n01440764/ …``  — the standard ImageNet layout; with all 1000 synset folders present, the class
  index is the position in sorted order (the torchvision ``ImageFolder`` convention).

Samples are listed in a deterministic order (class index, then file name), which makes
dataset generation reproducible and resumable by sample index.
"""
import re
from pathlib import Path
from typing import Optional, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGENET_NUM_CLASSES = 1000
CLASS_DIR = re.compile(r"^class_(\d+)$")


def discover_classes(root: Path) -> dict[str, int]:
    """Map every class folder below ``root`` to its ImageNet class index."""
    folders = sorted(d.name for d in root.iterdir() if d.is_dir())
    if folders and all(CLASS_DIR.match(name) for name in folders):
        return {name: int(CLASS_DIR.match(name).group(1)) for name in folders}
    if len(folders) == IMAGENET_NUM_CLASSES:
        return {name: index for index, name in enumerate(folders)}
    raise ValueError(
        f"{root}: expected 'class_<id>' folders or all {IMAGENET_NUM_CLASSES} ImageNet synset folders, "
        f"found {len(folders)} folders"
    )


def image_transform(image_size: int) -> T.Compose:
    """Resize and centre-crop to ``image_size``; returns uint8 ``(3, H, W)`` (normalised later on the device)."""
    return T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.PILToTensor()])


class ClassFolderImages(Dataset):
    def __init__(
        self,
        root: str,
        class_labels: Optional[Sequence[int]] = None,
        max_per_class: Optional[int] = None,
        image_size: int = 256,
    ):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Source image folder not found: {root}")
        classes = discover_classes(root)
        wanted = None if class_labels is None else set(class_labels)
        if wanted is not None and (missing := sorted(wanted - set(classes.values()))):
            raise ValueError(f"{root}: no folders for requested classes {missing}")

        self.samples: list[tuple[Path, int]] = []
        for folder, label in sorted(classes.items(), key=lambda item: item[1]):
            if wanted is not None and label not in wanted:
                continue
            files = sorted(p for p in (root / folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            self.samples.extend((path, label) for path in files[:max_per_class])
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label
