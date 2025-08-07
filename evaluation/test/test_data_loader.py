# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import os
import sys
import shutil
from pathlib import Path

# Standard library imports
import tempfile

import torch

import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image


# helper


# Jutils

# Setup project root for import resolution
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")
)
sys.path.append(project_root)


from data_processing.tools.norm import (
    denorm_tensor,
)  # denorm tensor -- just for plotting

torch.set_float32_matmul_precision("high")

from filelock import FileLock


import pytest


##############################################
# Dataset to wrap loaded samples
##############################################
class SavedSamplesDataset(Dataset):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.fake_dir = self.base_dir / "fake-images"
        self.real_dir = self.base_dir / "gt-images"
        self.label_dir = self.base_dir / "labels"

        self.file_list = self._collect_valid_triplets()

        print(f"[INFO] Found {len(self.file_list)} complete samples in {self.base_dir}")
        assert self.file_list, f"No complete sample triplets found in {self.base_dir}"

    def _is_valid_id(self, name: str) -> bool:
        return name.isdigit()

    def _collect_valid_triplets(self):
        fake_ids = {
            f.stem for f in self.fake_dir.glob("*.png") if self._is_valid_id(f.stem)
        }
        real_ids = {
            f.stem for f in self.real_dir.glob("*.png") if self._is_valid_id(f.stem)
        }
        label_ids = {
            f.stem for f in self.label_dir.glob("*.npy") if self._is_valid_id(f.stem)
        }
        valid_ids = sorted(fake_ids & real_ids & label_ids)
        return [f"{fid}.png" for fid in valid_ids]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        fake_path = self.fake_dir / file_name
        real_path = self.real_dir / file_name
        label_path = self.label_dir / file_name.replace(".png", ".npy")

        try:
            fake_tensor = load_png_to_tensor_normalized(fake_path)
            real_tensor = load_png_to_tensor_normalized(real_path)
            label = np.load(label_path)
            label_tensor = torch.tensor(
                label.item() if not np.isscalar(label) else label, dtype=torch.long
            )
            return fake_tensor, real_tensor, label_tensor

        except Exception as e:
            print(f"[WARNING] Failed to load sample {file_name}: {e}")
            return None  # Must be handled via custom collate_fn


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # drop empty batch

    collated = []
    for samples in zip(*batch):
        if isinstance(samples[0], torch.Tensor):
            collated.append(torch.stack(samples))
        else:
            collated.append(torch.tensor(samples))
    return tuple(collated)


def load_saved_samples_as_dataset(
    base_dir: str, batch_size: int = 8, shuffle: bool = False
):
    dataset = SavedSamplesDataset(base_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True,
        collate_fn=safe_collate,
    )


def save_tensor_as_png(tensor: torch.Tensor, path: str):
    """
    Save a torch tensor (C, H, W) as a .png image.
    Assumes tensor is in [-1, 1] or [0, 1].
    """
    tensor = denorm_tensor(tensor).detach().cpu()  # just for visualization
    img = TF.to_pil_image(tensor.byte())
    img.save(path)


def load_png_to_tensor(path: str) -> torch.Tensor:
    """
    Load a .png image and convert to torch.Tensor in [0, 1], shape [3, H, W].
    """
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # Returns float tensor in [0, 255]


def load_png_to_tensor_normalized(path: str) -> torch.Tensor:
    """
    Load .png image and return tensor in [-1, 1], shape [3, H, W].
    """
    tensor = load_png_to_tensor(path)  # [0, 255]
    tensor = tensor / 127.5 - 1.0  # [0, 255] → [-1, 1]
    return tensor


###############################################
# Utility function to clear a folder
###############################################
def clear_folder(folder_path: Path):
    lock_file = folder_path.with_suffix(".lock")
    with FileLock(lock_file):
        try:
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"[INFO] Removed existing folder: {folder_path}")
            elif folder_path.exists() and folder_path.is_file():
                folder_path.unlink()
                print(f"[INFO] Removed existing file: {folder_path}")
            else:
                print(f"[INFO] Folder {folder_path} does not exist, nothing to clear.")
        except Exception as e:
            print(f"[ERROR] Failed to clear folder {folder_path}: {e}")
            raise RuntimeError(f"[ERROR] Failed to clear folder {folder_path}: {e}")
        finally:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created folder: {folder_path}")


# ----------------------------------------
# Test Fixtures
# ----------------------------------------


def create_dummy_png(path, size=(3, 32, 32)):
    tensor = torch.rand(size) * 255
    img = TF.to_pil_image(tensor.byte())
    img.save(path)


def create_dummy_npy(path, value):
    np.save(path, np.array(value))


@pytest.fixture
def temp_sample_dir():
    tmpdir = tempfile.mkdtemp()
    base = Path(tmpdir)

    (base / "fake-images").mkdir()
    (base / "gt-images").mkdir()
    (base / "labels").mkdir()

    # Create 3 valid triplets
    for i in range(3):
        id_str = f"{i:07d}"
        create_dummy_png(base / "fake-images" / f"{id_str}.png")
        create_dummy_png(base / "gt-images" / f"{id_str}.png")
        create_dummy_npy(base / "labels" / f"{id_str}.npy", i)

    # Corrupted and unmatched cases
    (base / "labels" / "0000003.npy").write_text("corrupted")
    create_dummy_png(base / "fake-images" / "9999999.png")
    create_dummy_npy(base / "labels" / "8888888.npy", 8)

    yield base
    shutil.rmtree(base)


# ----------------------------------------
# Unit Tests
# ----------------------------------------


def test_dataset_triplet_matching(temp_sample_dir):
    ds = SavedSamplesDataset(temp_sample_dir)
    assert len(ds) == 3, "Should match only complete triplets"


def test_getitem_valid_sample(temp_sample_dir):
    ds = SavedSamplesDataset(temp_sample_dir)
    sample = ds[0]
    assert isinstance(sample[0], torch.Tensor)
    assert sample[0].shape == (3, 32, 32)
    assert isinstance(sample[2], torch.Tensor)
    assert sample[2].ndim == 0  # scalar tensor
    assert sample[2].dtype == torch.long


def test_getitem_corrupt_label_handled(temp_sample_dir):
    ds = SavedSamplesDataset(temp_sample_dir)
    ds.file_list.append("0000003.png")  # manually insert corrupted label
    sample = ds[-1]
    assert sample is None


def test_safe_collate_mixed_valid_and_none():
    valid = (torch.rand(3, 32, 32), torch.rand(3, 32, 32), torch.tensor(1))
    batch = [None, valid]
    out = safe_collate(batch)
    assert out is not None
    fake, real, label = out
    assert fake.shape[0] == 1
    assert isinstance(label, torch.Tensor)


def test_safe_collate_all_none():
    batch = [None, None]
    out = safe_collate(batch)
    assert out is None


def test_dataloader_with_valid_samples(temp_sample_dir):
    loader = load_saved_samples_as_dataset(temp_sample_dir, batch_size=2, shuffle=False)
    for batch in loader:
        assert batch is not None
        fake, real, label = batch
        assert fake.shape == (2, 3, 32, 32)
        assert real.shape == (2, 3, 32, 32)
        assert label.shape == (2,)
        assert isinstance(label, torch.Tensor)


# ----------------------------------------
# Optional CLI Runner
# ----------------------------------------

if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
