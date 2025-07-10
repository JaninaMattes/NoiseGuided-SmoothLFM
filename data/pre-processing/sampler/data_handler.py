from data_processing.tools.norm import denorm_tensor
from ldm.dataloader.dataloader.dataloader import WebDataModuleFromNumpy
import datetime
import gc
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, Tuple

import einops
from tqdm import tqdm

import h5py
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import webdataset as wds

import torch
import torchvision.transforms as T
import torch.nn.functional as F


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)


#############################################################
#                          Utils                           #
#############################################################

def img_to_grid(img, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and img.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        img = einops.rearrange(
            img, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        img = einops.rearrange(img, "b c h w -> " + to)
    return img


def un_normalize_img(img):
    """ Convert from [-1, 1] to [0, 255] """
    img = ((img * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return img


def normalize_img(img):
    """ Convert from [0, 255] to [-1, 1] """
    img = img.to(torch.float32) / 127.5 - 1
    return img


#############################################################
#                 Numpy File Handler                        #
#############################################################
class NumpyDataHandler:
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = Path(base_dir)
        self.create_directory(self.base_dir)

        # Set up logging
        self.get_logger()

    def create_or_extend_dataset(self, group, key, data, dtype, is_scalar=False):
        """Create or extend a dataset in the HDF5 group."""
        if key in group:
            # Extend dataset
            dataset = group[key]
            if dataset.shape[1:] != data.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for dataset {key}: existing shape {dataset.shape[1:]}, new data shape {data.shape[1:]}")
            dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
            dataset[-data.shape[0]:] = data
        else:
            # Create dataset
            if is_scalar:
                group.create_dataset(key, data=data, dtype=dtype)
            else:
                maxshape = (None,) + \
                    data.shape[1:] if data.ndim > 1 else (None,)
                group.create_dataset(
                    key, data=data, dtype=dtype, maxshape=maxshape, chunks=True)
        return group[key]

    def get_last_image_number(self, directory: Path) -> int:
        """Get the last image number in the directory."""
        try:
            existing_files = [f for f in directory.glob('*.png')]
            if not existing_files:
                return 0
            numbers = [int(f.stem) for f in existing_files]
            return max(numbers) if numbers else 0
        except Exception as e:
            logging.error(f"Error getting last image number: {e}")
            return 0

    def create_directory(self, path: Path) -> None:
        """Create directory if it does not exist."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory {path}")
        except PermissionError as e:
            logging.error(
                f"Permission denied when creating directory {path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error creating directory {path}: {e}")
            raise

    def save_to_numpy(self, data: Dict[str, Any], group_name: str = 'train') -> str:
        """
        Store images, labels, and latents in a structured directory format.
        >>> Example dataset structure:
                base_directory/ (e.g. train or validation)
                ├── Images/
                │   ├── 00001.png
                │   └── ...
                ├── Labels/
                │   ├── 00001.npy
                │   └── ...
                └── Latents_0.00/
                │   ├── 00001.npy
                │   └── ...
                └── ...
        """

        # Validate input data
        is_valid, error_msg = self.validate_dict(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {error_msg}")

        # Create group directory
        group_dir = self.base_dir / group_name
        img_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        self.create_directory(group_dir)
        self.create_directory(img_dir)
        self.create_directory(labels_dir)

        # Increment index
        start_index = self.get_last_image_number(img_dir) + 1
        print(f"[INFO] Starting index for saving: {start_index}")
        num_samples = len(data['image'])

        if 'image' in data:
            self.save_images(data['image'], img_dir, start_index)

        if 'label' in data:
            labels_dir = group_dir / 'Labels'
            self.create_directory(labels_dir)
            self.save_labels(data['label'], labels_dir, start_index)

        if 'recon_image' in data:
            recon_dir = group_dir / 'Recon_Images'
            self.create_directory(recon_dir)
            self.save_images(data['recon_image'], recon_dir, start_index)

        if 'latent' in data:
            latent_dir = group_dir / 'Latents_0.00'
            self.create_directory(latent_dir)
            self.save_latents(data['latent'], latent_dir, start_index)

        if 'intermediate_steps' in data and 'intermediates' in data:
            for step, interm_batch in zip(data['intermediate_steps'], data['intermediates']):
                if step == 0.0:  # Skip end
                    continue
                latent_dir = group_dir / f'Latents_{step:.2f}'
                self.create_directory(latent_dir)
                self.save_latents(interm_batch, latent_dir, start_index)

        self.store_metadata(group_dir, data, start_index, num_samples)

        logging.info(
            f"[INFO] Successfully saved batch of {num_samples} samples to {group_dir}")
        return str(group_dir)

    def save_latents(self, latents: np.ndarray, latent_dir: Path, start_index: int) -> None:
        """Save batch of latent representations as numpy files."""
        # Convert to numpy array
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()

        for i, latent in enumerate(latents):
            latent_path = latent_dir / f'{start_index + i:010d}.npy'
            try:
                # Store unnormalized with single precision
                latent = latent.astype(np.float32)
                np.save(str(latent_path), latent)
            except Exception as e:
                logging.error(f"Error saving latent {latent_path}: {e}")
                raise

    def save_images(self, images: np.ndarray, img_dir: Path, start_index: int) -> None:
        """Save batch of original images in [-1,1] normalised form as png files."""
        # Convert to numpy array
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()

        if isinstance(images, np.ndarray):
            images = images.transpose(0, 2, 3, 1)
            images = torch.from_numpy(images)

        images = denorm_tensor(images)          # to [0, 255]
        transform = T.ToPILImage()

        for i, img in enumerate(images):
            image_path = img_dir / f'{start_index + i:010d}.png'
            try:
                # store unnormalized
                img = transform(img)
                img.save(str(image_path))
            except Exception as e:
                logging.error(f"Error saving image {image_path}: {e}")
                raise

    def save_labels(self, labels: np.ndarray, labels_dir: Path, start_index: int) -> None:
        """Save labels as individual numpy files with double precision."""
        labels = labels.detach().cpu().numpy()
        for i, label in enumerate(labels):
            label_path = labels_dir / f'{start_index + i:010d}.npy'
            try:
                # Double precision
                label = label.astype(np.float64)
                np.save(str(label_path), label)
            except Exception as e:
                logging.error(f"Error saving label {label_path}: {e}")
                raise

    def load_images(self, img_dir: Path, start_index: int, num_images: int) -> np.ndarray:
        """Load batch of images and convert them back to [-1,1] normalised form."""
        images = []
        for i in range(num_images):
            image_path = img_dir / f'{start_index + i:010d}.png'
            try:
                img_data = plt.imread(str(image_path))
                # Correct format (C, H, W)
                img_data = img_data.transpose(2, 0, 1)
                # Normalize from [0, 255] to [-1, 1]
                img_data = normalize_img(img_data).astype(np.float32)
                images.append(img_data)
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                raise
        return np.array(images)

    def load_latents(self, latent_dir: Path, start_index: int, num_images: int) -> np.ndarray:
        """Load batch of latent representations."""
        latents = []
        for i in range(num_images):
            latent_path = latent_dir / f'{start_index + i:010d}.npy'
            try:
                # TODO: Needs fix
                latent_data = np.load(str(latent_path))
                # Correct format (C, H, W)
                latent_data = latent_data.transpose(2, 0, 1).astype(np.float32)
                latents.append(latent_data)
            except Exception as e:
                logging.error(f"Error loading latent {latent_path}: {e}")
                raise
        return np.array(latents)

    def load_labels(self, labels_dir: Path, start_index: int, num_images: int) -> np.ndarray:
        """Load batch of labels."""
        labels = []
        for i in range(num_images):
            label_path = labels_dir / f'{start_index + i:010d}.npy'
            try:
                # TODO: Needs fix
                # Load labels as long double
                label_data = np.load(str(label_path)).astype(np.float64)
                labels.append(label_data)
            except Exception as e:
                logging.error(f"Error loading label {label_path}: {e}")
                raise
        return np.array(labels)

    def load_from_numpy(
        self,
        base_dir: Path,
        batch_size: int = 16
    ) -> wds.WebLoader:
        """ Load images, labels, and latents from numpy files """
        loader = WebDataModuleFromNumpy(
            base_dir=base_dir,
            batch_size=batch_size
        )
        return loader

    """ Utility functions """

    def store_metadata(self, group_dir: Path, data: Dict[str, Any], start_index: int, num_samples: int) -> None:
        """Save metadata about the stored data."""
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'start_index': str(start_index),
            'num_samples': str(num_samples),
            'image_shape': data['image'].shape[1:] if 'image' in data else None,
            'latent_shape': data['latent'].shape[1:] if 'latent' in data else None,
            'intermediate_steps': data.get('intermediate_steps', []),
            'has_labels': 'label' in data
        }

        metadata_path = os.path.join(group_dir, 'metadata.json')

        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    try:
                        existing_metadata = json.load(f)
                    except json.JSONDecodeError:
                        existing_metadata = []  # overwrite existing metadata
                if isinstance(existing_metadata, dict):
                    existing_metadata = [existing_metadata]  # convert to list
                existing_metadata.append(metadata)
                metadata = existing_metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")

    def get_logger(self) -> None:
        """Configure logging with detailed format."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_dir / 'storage.log')
            ]
        )

    def validate_dict(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate data dictionary before saving."""
        required_keys = ['image']
        for key in required_keys:
            if key not in data:
                return False, f"Missing required key: {key}"

        if 'intermediates' in data and 'intermediate_steps' not in data:
            return False, "intermediate_steps required when intermediates present"

        if 'intermediate_steps' in data and 'intermediates' in data:
            if len(data['intermediate_steps']) != len(data['intermediates']):
                return False, "Mismatch between steps and intermediates length"

        return True, "No errors"


#############################################################
#                HDF5 File Handler                        #
#############################################################
# -new version-

class HDF5DatasetManager:
    """ 
    HDF5 Dataset Manager from Numpy + PNG Files 
    Example structure of stored data:

    base_directory/
    ├── train/
        ├── images/
        ├── label/
        ├── latent_<timestep>/
    ├── validation/
        ├── images/
        ├── label/
        ├── latent_<timestep>/
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    """ Utility functions """

    def create_or_extend_dataset(self, group, dataset_name, data, dtype=np.float32, is_scalar=False):
        """Create or extend a dataset in the HDF5 group."""
        if dataset_name in group:
            dataset = group[dataset_name]
            dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
            dataset[-data.shape[0]:] = data
        else:
            if is_scalar:
                group.create_dataset(dataset_name, data=data, dtype=dtype)
            else:
                group.create_dataset(
                    dataset_name, data=data, dtype=dtype, maxshape=(None,) + data.shape[1:])

        print(f"Dataset {dataset_name} has been created or extended.")
        return group[dataset_name]

    """    Save to HDF5 (New Version)   """

    def save_to_hdf5(self,
                     group_name='train',
                     img_shape=(3, 256, 256),
                     latent_shape=(4, 32, 32),
                     label_shape=(1,),
                     filename='data.hdf5',
                     timesteps=None
                     ):
        """ Save images, labels, and latents to HDF5 file.
            Example required dataset structure:
                base_directory/ (e.g. train or validation)
                ├── Images/
                │   ├── 00001.png
                │   └── ...
                ├── Labels/
                │   ├── 00001.npy
                │   └── ...
                └── Latents_0.00/
                │   ├── 00001.npy
                │   └── ...
                └── ...
        """

        file_path = os.path.join(self.base_dir, filename)
        print(f"[INFO] Saving data to HDF5 file: {file_path}")

        group_dir = Path(self.base_dir) / group_name
        images_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        # Check if directories exist
        if timesteps is not None:
            timestep_names = {f"Latents_{t:.2f}" for t in timesteps}
            latent_timestep_dirs = [d for d in group_dir.glob(
                'Latents_*') if d.name in timestep_names]
        else:
            latent_timestep_dirs = sorted(group_dir.glob('Latents_*'))

        if not images_dir.exists() or not labels_dir.exists():
            print(
                f"[ERROR] Required directories (Images/Labels) do not exist in {group_dir}")
            return None

        print(
            f"[INFO] Found {len(latent_timestep_dirs)} latent directories: {latent_timestep_dirs}")

        # Create HDF5 file and group
        with h5py.File(file_path, 'a') as h5_file:
            group = h5_file.require_group(group_name)
            image_paths = sorted(images_dir.glob(
                '*.png'), key=lambda p: p.stem)
            label_paths = sorted(labels_dir.glob(
                '*.npy'), key=lambda p: p.stem)

            if len(image_paths) == 0 or len(label_paths) == 0:
                print(
                    f"[ERROR] No image or label files found in {images_dir} or {labels_dir}")
                return None

            # Ensure images and labels match
            print(
                f"[INFO] Found {len(image_paths)} images and {len(label_paths)} labels in {group_name} group.")

            if len(image_paths) != len(label_paths):
                self.retrieve_missing_files(
                    image_paths, label_paths, group_name)
            assert len(image_paths) == len(
                label_paths), "Number of images and labels do not match!"

            num_samples = len(image_paths)
            img_dset = group.create_dataset(
                'images', shape=(0,) + img_shape, maxshape=(None,) + img_shape,
                dtype=np.float32, chunks=(1,) + img_shape, compression="gzip"
            )
            lbl_dset = group.create_dataset(
                'labels', shape=(0,) + label_shape, maxshape=(None,) + label_shape,
                dtype=np.int64, chunks=(1,) + label_shape, compression="gzip"
            )

            for img_path, label_path in tqdm(zip(image_paths, label_paths), desc="Processing images and labels", total=num_samples):
                # Check if image and label file names match
                if img_path.stem != label_path.stem:
                    print(
                        f"[ERROR] Mismatched file names: {img_path.name} <-> {label_path.name}")
                    continue

                try:
                    # Load and validate image
                    with Image.open(img_path) as im:
                        img = im.convert('RGB')
                        img = np.asarray(img, dtype=np.float32)

                    if img.shape != (256, 256, 3):  # or your expected shape
                        print(
                            f"[WARNING] Unexpected image shape: {img.shape} for file: {img_path}")
                        continue

                    img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)

                    # Load and validate label
                    label = np.load(label_path).astype(np.int64).reshape(-1)

                except Exception as e:
                    print(
                        f"[ERROR] Failed to load image or label:\n  Image: {img_path}\n  Label: {label_path}\n  Reason: {e}")
                    raise ValueError(f"Failed to load image or label: {e}")

                img_dset.resize(img_dset.shape[0] + 1, axis=0)
                lbl_dset.resize(lbl_dset.shape[0] + 1, axis=0)
                img_dset[-1] = img
                lbl_dset[-1] = label

            for timestep_dir in tqdm(latent_timestep_dirs, desc="Processing latents"):
                # Check if directory name matches expected format
                if not timestep_dir.name.startswith('Latents_'):
                    print(
                        f"[ERROR] Invalid directory name: {timestep_dir.name}")
                    continue
                timestep = timestep_dir.name.split('_')[-1]
                latent_group_name = f'latents_{timestep}'

                if latent_group_name not in group:
                    lat_dset = group.create_dataset(
                        latent_group_name, shape=(0,) + latent_shape, maxshape=(None,) + latent_shape,
                        dtype=np.float32, chunks=(1,) + latent_shape, compression="gzip"
                    )
                else:
                    lat_dset = group[latent_group_name]

                latent_paths = sorted(timestep_dir.glob('*.npy'))
                if len(latent_paths) == 0:
                    print(
                        f"[ERROR] No latent files found for timestep {timestep}")
                    continue

                for latent_path in tqdm(latent_paths, desc=f"Processing timestep {timestep}", leave=False):
                    latent = np.load(latent_path).astype(np.float32)

                    if latent is None:
                        continue

                    lat_dset.resize(lat_dset.shape[0] + 1, axis=0)
                    lat_dset[-1] = latent

        print(f"Data has been stored to {file_path}.")
        return file_path

    def append_to_hdf5(self,
                       group_name='train',
                       img_shape=(3, 256, 256),
                       latent_shape=(4, 32, 32),
                       label_shape=(1,),
                       filename='data.hdf5',
                       timesteps=None):
        """
        Append new samples to an existing HDF5 dataset.
        """
        file_path = os.path.join(self.base_dir, filename)
        print(f"[INFO] Appending data to HDF5 file: {file_path}")

        group_dir = Path(self.base_dir) / group_name
        images_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        if timesteps is not None:
            timestep_names = {f"Latents_{t:.2f}" for t in timesteps}
            latent_timestep_dirs = [d for d in group_dir.glob(
                'Latents_*') if d.name in timestep_names]
        else:
            latent_timestep_dirs = sorted(group_dir.glob('Latents_*'))
        print(
            f"[INFO] Found {len(latent_timestep_dirs)} latent directories: {latent_timestep_dirs}")

        if not images_dir.exists() or not labels_dir.exists():
            print(
                f"[ERROR] Required directories (Images/Labels) do not exist in {group_dir}")
            return None

        with h5py.File(file_path, 'a') as h5_file:
            if group_name not in h5_file:
                raise ValueError(
                    f"[ERROR] Group '{group_name}' not found in {file_path}. Run save_to_hdf5 first.")

            group = h5_file[group_name]
            image_paths = sorted(images_dir.glob(
                '*.png'), key=lambda p: p.stem)
            label_paths = sorted(labels_dir.glob(
                '*.npy'), key=lambda p: p.stem)

            if len(image_paths) != len(label_paths):
                self.retrieve_missing_files(
                    image_paths, label_paths, group_name)
            assert len(image_paths) == len(
                label_paths), "Number of images and labels do not match!"

            num_samples = len(image_paths)

            if 'images' not in group or 'labels' not in group:
                raise ValueError(
                    "[ERROR] Missing 'images' or 'labels' dataset in the HDF5 group.")

            img_dset = group['images']
            lbl_dset = group['labels']

            for img_path, label_path in tqdm(zip(image_paths, label_paths), desc="Appending images and labels", total=num_samples):
                if img_path.stem != label_path.stem:
                    print(
                        f"[ERROR] Mismatched file names: {img_path.name} <-> {label_path.name}")
                    continue

                try:
                    with Image.open(img_path) as im:
                        img = im.convert('RGB')
                        img = np.asarray(img, dtype=np.float32)

                    if img.shape != (256, 256, 3):
                        print(f"[WARNING] Unexpected image shape: {img.shape}")
                        continue

                    img = np.transpose(img, (2, 0, 1))
                    label = np.load(label_path).astype(np.int64).reshape(-1)

                except Exception as e:
                    print(f"[ERROR] Failed to load image or label: {e}")
                    continue

                img_dset.resize(img_dset.shape[0] + 1, axis=0)
                lbl_dset.resize(lbl_dset.shape[0] + 1, axis=0)
                img_dset[-1] = img
                lbl_dset[-1] = label

            for timestep_dir in tqdm(latent_timestep_dirs, desc="Appending latents"):
                if not timestep_dir.name.startswith('Latents_'):
                    continue
                timestep = timestep_dir.name.split('_')[-1]
                latent_group_name = f'latents_{timestep}'

                if latent_group_name not in group:
                    print(
                        f"[WARNING] Latent group '{latent_group_name}' not found in HDF5. Skipping.")
                    continue

                lat_dset = group[latent_group_name]
                latent_paths = sorted(timestep_dir.glob('*.npy'))

                for latent_path in tqdm(latent_paths, desc=f"Appending timestep {timestep}", leave=False):
                    try:
                        latent = np.load(latent_path).astype(np.float32)
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to load latent file {latent_path}: {e}")
                        continue

                    lat_dset.resize(lat_dset.shape[0] + 1, axis=0)
                    lat_dset[-1] = latent

        print(f"[INFO] Appended data to {file_path}.")
        return file_path

    def retrieve_missing_files(self, image_paths, label_paths, group_name='train'):
        """
        Check for missing image and label files in the specified group.
        This function compares the stems of image and label files to ensure they match.
        If there are mismatches, it prints warnings and returns without proceeding further.
        """
        # === Consistency check: matching stems ===
        image_stems = {p.stem for p in image_paths}
        label_stems = {p.stem for p in label_paths}

        missing_images = label_stems - image_stems
        missing_labels = image_stems - label_stems

        if missing_images:
            print(
                f"[WARNING] Missing PNGs for {len(missing_images)} label(s): {[f + '.png' for f in sorted(missing_images)]}")
        if missing_labels:
            print(
                f"[WARNING] Missing NPYs for {len(missing_labels)} image(s): {[f + '.npy' for f in sorted(missing_labels)]}")

        return missing_images, missing_labels
        # If there are any missing files, print an error and return

    def retrieve_from_hdf5(self, file_path: str, group_name: str, timestep: float, batch_size: int = 32, start_idx: int = 0, plot_samples: bool = False):
        try:
            with h5py.File(file_path, 'r') as h5_file:
                group = h5_file[group_name]

                images = group['images'][start_idx:start_idx+batch_size]
                labels = group['labels'][start_idx:start_idx+batch_size]
                latent_key = f'latents_{timestep:.2f}'
                latents = group[latent_key][start_idx:start_idx+batch_size]

            if plot_samples:
                num_samples = min(8, len(images))
                fig, axes = plt.subplots(
                    2, num_samples, figsize=(2 * num_samples, 4))

                for i in range(num_samples):
                    img = images[i].transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f"Label: {labels[i][0]}")
                    axes[0, i].axis("off")

                    latent_img = latents[i][:3, :, :].transpose(1, 2, 0)
                    latent_img = (latent_img - latent_img.min()) / \
                        (latent_img.max() - latent_img.min())
                    axes[1, i].imshow(latent_img)
                    axes[1, i].set_title("Latent Space")
                    axes[1, i].axis("off")

                plt.tight_layout()
                output_file = os.path.join(
                    self.base_dir, f'dummy_{group_name}_data.png')
                plt.savefig(output_file)
                print(f"[INFO] Saved sample plot to {output_file}")
                plt.show()
                plt.close(fig)

            return images, labels, latents

        except Exception as e:
            print(f"[ERROR] During HDF5 data retrieval: {str(e)}")
            gc.collect()
            raise

    def print_hdf5_structure(self, file_path: str, save_to_file: bool = False, output_file: str = "hdf5_structure.txt"):
        """Print the structure of an HDF5 file in a tree-like format and optionally save it to a file."""
        lines = []  # Store lines for optional file writing

        def print_structure(name, obj, depth=0, last=False, prefix=""):
            indent = "    " * (depth - 1) + \
                ("└── " if last else "├── ") if depth > 0 else ""
            line = f"{prefix}{indent}{name.split('/')[-1]}"
            if isinstance(obj, h5py.Dataset):
                line += f" (Dataset, shape={obj.shape}, dtype={obj.dtype})"
            lines.append(line)

            if isinstance(obj, h5py.Group):
                children = list(obj.keys())
                for i, child in enumerate(children):
                    print_structure(
                        f"{name}/{child}", obj[child], depth + 1, i == len(children) - 1, prefix)

        # Open the HDF5 file and print its structure
        with h5py.File(file_path, 'r') as h5_file:
            lines.append(f"{file_path}")  # Root node
            print_structure("/", h5_file)

        # Plot the structure
        for line in lines:
            print(line)

        # Save to file if requested
        if save_to_file:
            output_file = os.path.join(self.base_dir, output_file)
            with open(output_file, "w") as f:
                f.write("\n".join(lines))
            print(f"Structure saved to {output_file}")

    """ Sanity check """
    @staticmethod
    def create_dummy_dataset(base_dir='dummy_data', group_name='train', samples=5):
        # Create small dummy dataset for testing
        os.makedirs(base_dir, exist_ok=True)

        train_dir = os.path.join(base_dir, group_name)
        images_dir = os.path.join(train_dir, 'Images')
        labels_dir = os.path.join(train_dir, 'Labels')
        latents_dir_0 = os.path.join(train_dir, 'Latents_0.1')
        latents_dir_25 = os.path.join(train_dir, 'Latents_0.2')

        for d in [images_dir, labels_dir, latents_dir_0, latents_dir_25]:
            os.makedirs(d, exist_ok=True)

        # Create some dummy data
        for i in range(samples):
            img = (np.random.rand(256, 256, 3) *
                   255).astype(np.uint8)      # Random image
            # Random label between 0 and 9
            label = np.random.randint(0, 10, size=(1,), dtype=np.int64)
            latent = np.random.rand(4, 32, 32).astype(np.float32)

            Image.fromarray(img).save(os.path.join(images_dir, f"{i}.png"))
            np.save(os.path.join(labels_dir, f"{i}.npy"), label)
            np.save(os.path.join(latents_dir_0, f"{i}.npy"), latent)
            np.save(os.path.join(latents_dir_25, f"{i}.npy"), latent)

        print(f"[Warning] Dummy dataset created in {base_dir}...")


""" Store data to HDF5 """


def save_hdf5(data_dir, filename, timestep=0.5, group_name: str = 'train', timesteps=None):
    """ Save data to HDF5 file. """
    # Store data
    hdfhandler = HDF5DatasetManager(data_dir)
    hdfhandler.save_to_hdf5(
        filename=filename, group_name=group_name,  timesteps=timesteps)

    # Show structure
    hdf5_file = os.path.join(data_dir, filename)
    print(f"Filepath created {hdf5_file}")

    hdfhandler.print_hdf5_structure(hdf5_file, save_to_file=True)

    # Load HDF5
    imgs, labels, latents = hdfhandler.retrieve_from_hdf5(
        file_path=hdf5_file, timestep=timestep, group_name=group_name, plot_samples=True)

    print(f"Done - Filepath used: {hdf5_file}")
    return imgs, labels, latents


def extend_to_hdf5(data_dir, filename, timesteps=None, group_name='train'):
    """ Save data to HDF5 file. """
    # Store data
    hdfhandler = HDF5DatasetManager(data_dir)

    # Pass the timesteps if provided
    hdfhandler.extend_to_hdf5(
        filename=filename, group_name=group_name, timesteps=timesteps)

    # Show structure
    hdf5_file = os.path.join(data_dir, filename)
    print(f"Filepath created {hdf5_file}")

    # Optionally print the HDF5 structure
    hdfhandler.print_hdf5_structure(hdf5_file, save_to_file=True)

    # Load HDF5
    imgs, labels, latents = hdfhandler.retrieve_from_hdf5(
        file_path=hdf5_file, timestep=timesteps, group_name=group_name, plot_samples=True)

    print(f"Done - Filepath used: {hdf5_file}")
    return imgs, labels, latents


def check_and_update_latents(existing_lat_count, latent_paths, existing_lat_files, lat_dset):
    """
    Check if all latent files exist in the HDF5 file and update only the new ones.
    """
    existing_lat_filenames = [os.path.basename(
        file) for file in existing_lat_files]
    new_latent_files = [latent for latent in latent_paths if os.path.basename(
        latent) not in existing_lat_filenames]

    # If there are new latent files, process them
    if new_latent_files:
        print(f"Found {len(new_latent_files)} new latent files.")
        for latent in new_latent_files:
            print(f"Adding latent: {latent}")
            # Load latent and add it to the dataset
            latent_data = np.load(latent).astype(np.float32)
            # Resize dataset to accommodate new latent
            lat_dset.resize(lat_dset.shape[0] + 1, axis=0)
            lat_dset[-1] = latent_data
    else:
        print("No new latent files to add.")



def save_hdf5_intermediate_latents(data_dir, filename, timestep=0.5, group_name='train', timesteps=None):
    """Save only latents to HDF5 file, then retrieve for inspection."""

    hdfhandler = HDF5DatasetManager(data_dir)
    # Save specific latents only
    hdfhandler.save_intermediate_latents(
        group_name=group_name, filename=filename, latent_shape=(4, 32, 32), timesteps=timesteps)

    # Show file structure after save
    hdf5_file = os.path.join(data_dir, filename)
    hdfhandler.print_hdf5_structure(hdf5_file, save_to_file=True)
    print(f"Data saved to {hdf5_file}")

    # Load and return saved latents for verification
    imgs, labels, latents = hdfhandler.retrieve_from_hdf5(
        file_path=hdf5_file, timestep=timestep, group_name=group_name, plot_samples=True)

    return imgs, labels, latents








if __name__ == "__main__":

    ########################################
    ################ Folders ###############
    ########################################
    hdf5_file = "./dataset/processed/imagenet-256/"
    data_dir = "./dataset/processed/imagenet-256/"

    # 'train' or 'validation'
    group_name = 'train'
    # select for test plot
    sample_timestep = 0.50
    # timesteps = [0.90, 1.00]                                    # Optional: Select timesteps for saving latents only

    ########################################
    ############### Filename ###############
    ########################################

    postfix = datetime.datetime.now().strftime("T%H%M%S")
    filename = f'imagenet256_data_set-{postfix}.hdf5'
    filename = "imagenet256_data.hdf5"
    "imagenet256_data.hdf5" / 'imagenet256_data-T200425.hdf5'

    ########################################
    ######### Saving Procedure #############
    ########################################
    print(f"Filename set to: {filename}")
    print(f"Data directory set to: {data_dir}")
    print(f"Group name set to: {group_name}")
    print(f"Sample timestep set to: {sample_timestep}")

    # imgs, labels, latents = extend_to_hdf5(data_dir, filename, sample_timestep, group_name)
    imgs, labels, latents = save_hdf5(
        data_dir, filename, sample_timestep, group_name)   # All files

    # imgs, labels, latents = save_hdf5_intermediate_latents(data_dir, 
    # filename, timestep=sample_timestep, group_name=group_name, timesteps=timesteps)      # Only some files not all

    # Print shapes
    print(
        f"Images: {imgs.shape}, Labels: {labels.shape}, Latents: {latents.shape}")
    print(f"Images min: {imgs.min()}, max: {imgs.max()}")
    print(f"Latents min: {latents.min()}, max: {latents.max()}")


# CUDA_VISIBLE_DEVICES=0 python ...
