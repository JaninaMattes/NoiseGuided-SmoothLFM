"""Reader for the guidance-free HDF5 dataset written by ``dataprocessing.guidance_free``.

File layout (one group per split, e.g. ``train``, ``validation``, ``test``)::

    <split>/images          uint8 or float32  (N, 3, 256, 256)  pixel values in [0, 255]
    <split>/labels          int64             (N, 1)            ImageNet class index
    <split>/latents_{t:.2f} float32           (N, 4, 32, 32)    x_t on the forward ODE path (t=1 clean, t=0 noise)

Each sample is a dict with ``image`` in [-1, 1], ``label`` and the two requested latents
``latents_{source:.2f}`` / ``latents_{target:.2f}``, the keys the trainers read.
"""
import logging
import os
from typing import Optional

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def latent_key(timestep: float) -> str:
    return f"latents_{timestep:.2f}"


class HDF5LatentsDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        group_name: str,
        source_timestep: float,
        target_timestep: Optional[float] = None,
        p: float = 0.0,
    ):
        self.hdf5_file = os.path.abspath(hdf5_file)
        self.group_name = group_name
        self.p = p                                  # probability of a horizontal flip
        self.keys = [latent_key(source_timestep)]
        if target_timestep is not None and latent_key(target_timestep) not in self.keys:
            self.keys.append(latent_key(target_timestep))

        with h5py.File(self.hdf5_file, "r") as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.hdf5_file} (has: {list(f)})")
            group = f[group_name]
            missing = [k for k in ["images", "labels", *self.keys] if k not in group]
            if missing:
                raise ValueError(f"Datasets {missing} not found in '{group_name}' of {self.hdf5_file}")
            self.length = len(group["images"])
        self._file = None                           # opened lazily, once per DataLoader worker

    def __len__(self) -> int:
        return self.length

    def _group(self) -> h5py.Group:
        if self._file is None:
            self._file = h5py.File(self.hdf5_file, "r")
        return self._file[self.group_name]

    def labels(self) -> np.ndarray:
        with h5py.File(self.hdf5_file, "r") as f:
            return f[self.group_name]["labels"][()].reshape(-1).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        group = self._group()
        image = torch.from_numpy(np.asarray(group["images"][idx], dtype=np.float32)) / 127.5 - 1.0
        sample = {
            "image": image,
            "label": torch.tensor(int(np.asarray(group["labels"][idx]).reshape(-1)[0]), dtype=torch.long),
            **{key: torch.from_numpy(np.asarray(group[key][idx], dtype=np.float32)) for key in self.keys},
        }
        if self.p > 0 and torch.rand(1).item() < self.p:
            sample = {k: torch.flip(v, dims=[-1]) if v.ndim == 3 else v for k, v in sample.items()}
        return sample


def class_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Draw every class with equal probability (inverse class-frequency weights)."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(1.0 / counts[inverse], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class HDF5DataModule(LightningDataModule):
    def __init__(
        self,
        hdf5_file: str,
        source_timestep: float,
        target_timestep: Optional[float] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        train_group: Optional[str] = "train",
        val_group: Optional[str] = "validation",
        test_group: Optional[str] = None,
        num_workers: int = 8,
        val_num_workers: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        balance_classes: bool = True,
        p: float = 0.5,
        drop_last: bool = False,
        seed: int = 2025,
    ):
        super().__init__()
        self.hdf5_file = hdf5_file
        self.timesteps = dict(source_timestep=source_timestep, target_timestep=target_timestep)
        self.groups = dict(train=train_group, validation=val_group, test=test_group)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.val_num_workers = num_workers if val_num_workers is None else val_num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.balance_classes = balance_classes
        self.p = p
        self.drop_last = drop_last
        self.seed = seed
        self.datasets: dict[str, HDF5LatentsDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        for split, group in self.groups.items():
            if group is not None and split not in self.datasets:
                flip = self.p if split == "train" else 0.0    # augment training data only
                self.datasets[split] = HDF5LatentsDataset(self.hdf5_file, group, p=flip, **self.timesteps)
                logger.info(f"{split}: {len(self.datasets[split])} samples from group '{group}'")

    def _loader(self, split: str, batch_size: int, num_workers: int, train: bool) -> DataLoader:
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' is not available - set {split}_group and call setup().")
        dataset = self.datasets[split]
        sampler = class_balanced_sampler(dataset.labels()) if train and self.balance_classes else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=self.drop_last if train else False,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", self.batch_size, self.num_workers, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("validation", self.val_batch_size, self.val_num_workers, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", self.val_batch_size, self.val_num_workers, train=False)
