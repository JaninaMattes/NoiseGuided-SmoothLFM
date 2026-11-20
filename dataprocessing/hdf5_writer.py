"""Append-only writer for the guidance-free HDF5 dataset.

Writes the layout read by ``ldm.dataloader.dataloader.hdf5_dataloader``::

    /<split>/images            uint8   (N, 3, H, W)
    /<split>/labels            int64   (N, 1)
    /<split>/latents_{t:.2f}   float32 (N, 4, h, w)

Generation settings are stored as file attributes. Re-opening a file with different settings
is refused (provenance such as the git commit is recorded but not compared), so a resumed run can never mix two experimental setups in one dataset. If a run
was interrupted in the middle of a batch, all datasets of the split are truncated to the
longest complete prefix.
"""
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from ldm.dataloader.dataloader.hdf5_dataloader import latent_key   # single source of the key format

FORMAT_VERSION = 1


class HDF5DatasetWriter:
    def __init__(
        self,
        path: str,
        settings: dict[str, Any],
        timesteps: list[float],
        image_shape: tuple[int, ...] = (3, 256, 256),
        latent_shape: tuple[int, ...] = (4, 32, 32),
        compression: Optional[str] = "lzf",
        provenance: Optional[dict[str, Any]] = None,
    ):
        self.path = Path(path)
        self.settings = {"format_version": FORMAT_VERSION, **settings}
        self.timesteps = [float(t) for t in timesteps]
        self.shapes = {"images": image_shape, "labels": (1,), **{latent_key(t): latent_shape for t in self.timesteps}}
        self.dtypes = {"images": np.uint8, "labels": np.int64, **{latent_key(t): np.float32 for t in self.timesteps}}
        self.compression = compression
        self.provenance = provenance or {}
        self.file: Optional[h5py.File] = None

    def __enter__(self) -> "HDF5DatasetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "a")
        self._check_settings()
        return self

    def __exit__(self, *exc) -> None:
        self.file.close()
        self.file = None

    def _check_settings(self) -> None:
        encoded = {key: json.dumps(value, sort_keys=True) for key, value in self.settings.items()}
        if not self.file.attrs:
            self.file.attrs.update(encoded)
            self.file.attrs.update({f"provenance.{k}": json.dumps(v) for k, v in self.provenance.items()})
            return
        stored = dict(self.file.attrs)
        changed = sorted(k for k in encoded if stored.get(k) != encoded[k])
        if changed:
            raise ValueError(
                f"{self.path} was generated with different settings ({', '.join(changed)}). "
                "Use a new dataset_name or delete the file."
            )

    def _split(self, split: str) -> h5py.Group:
        group = self.file.require_group(split)
        for key, shape in self.shapes.items():
            if key not in group:
                group.create_dataset(
                    key, shape=(0, *shape), maxshape=(None, *shape), dtype=self.dtypes[key],
                    chunks=(1, *shape), compression=self.compression,
                )
        return group

    def num_samples(self, split: str) -> int:
        """Complete samples of ``split``; repairs a partially written last batch."""
        if split not in self.file:
            return 0
        group = self._split(split)
        complete = min(len(group[key]) for key in self.shapes)
        for key in self.shapes:
            if len(group[key]) != complete:
                group[key].resize(complete, axis=0)
        return complete

    def append(self, split: str, images: np.ndarray, labels: np.ndarray, latents: dict[float, np.ndarray]) -> None:
        arrays = {
            "images": images,
            "labels": np.asarray(labels).reshape(-1, 1),
            **{latent_key(t): latents[t] for t in self.timesteps},
        }
        size = len(images)
        for key, array in arrays.items():
            if array.shape != (size, *self.shapes[key]):
                raise ValueError(f"{key}: expected shape {(size, *self.shapes[key])}, got {array.shape}")
        group = self._split(split)
        start = self.num_samples(split)
        for key, array in arrays.items():
            group[key].resize(start + size, axis=0)
            group[key][start:] = array.astype(self.dtypes[key], copy=False)
        self.file.flush()
