
from data_processing.sampler.data_handler import NumpyDataHandler, HDF5DatasetManager
from data_processing.tools.norm import denorm_tensor
from ldm.models.transformer.dit import DiT_models
from ldm.flow import FlowModel
import os
import gc
import re
import sys
import json

import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from PIL import Image
from matplotlib import pyplot as plt
import einops

from jutils.nn import AutoencoderKL
from jutils import exists, freeze, default, ims_to_grid, instantiate_from_config, denorm
from jutils.vision import tensor2im

project_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)


# Seed
torch.manual_seed(2025)


torch.set_float32_matmul_precision('high')


# ---------- Data Loader ----------

class ClassIDImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[T.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for class_folder in self.root_dir.iterdir():
            if not class_folder.is_dir():
                continue
            match = re.match(r'class_(\d+)', class_folder.name)
            if not match:
                continue
            label = int(match.group(1))
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------- Image Utilities ----------

def img_to_grid(img: torch.Tensor, stack: str = "row", split: Optional[int] = 4) -> torch.Tensor:
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


def un_normalize_img(img: torch.Tensor) -> torch.Tensor:
    return ((img * 127.5) + 127.5).clamp(0, 255).to(torch.uint8)


def normalize_img(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 127.5 - 1


def show_samples(intermediates: Dict[str, torch.Tensor], save_dir: Optional[str] = None, prefix: str = "",
                 title: str = "One-sided diffusion") -> None:
    sorted_intermediates = dict(
        sorted(intermediates.items(), key=lambda x: float(x[0]), reverse=True))
    all_steps = torch.stack(
        list(sorted_intermediates.values()))  # [T, B, C, H, W]
    all_steps = all_steps.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]

    for i, sample in enumerate(all_steps):
        sample = denorm_tensor(sample)
        grid = make_grid(sample, nrow=sample.shape[0], padding=0)

        plt.figure(figsize=(sample.shape[0] * 2, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title(title)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"{prefix}_sample_{i}.png")
            plt.savefig(filename, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()
        plt.close()

        torch.cuda.empty_cache()
        gc.collect()


# ---------- Sample Processor ----------

class SampleProcessor:
    def __init__(
        self,
        dataloader: DataLoader,
        first_stage_ckpt: str,                      # First stage model checkpoint
        second_stage_ckpt: str,                     # Second stage model checkpoint
        source_data_dir: str,                       # original/raw data location
        target_data_dir: str,                       # where to save processed samples
        # where to save hdf5 files, defaults to target_data_dir
        hdf5_dir: Optional[str] = None,
        # HDF5 file name, defaults to imagenet256-testset-T%H%M%S.hdf5
        hdf5_file_name: Optional[str] = None,
        # Starting index for saving files
        start_file_idx: Optional[int] = None,
        selected_timesteps: List[float] = None,
        input_size: int = 32,
        num_classes: int = 1000,
        class_labels: List[int] = None,
        batch_size: int = 32,
        num_steps: int = 100,
        sample_kwargs: Optional[Dict] = None,
        device: Optional[str] = 'cuda',
        log_every: int = 1000,
        run_type: str = 'train',
        end_batch_id: int = 999999
    ):
        self.device = device
        if not torch.cuda.is_available() and self.device == 'cuda':
            raise ValueError(
                "CUDA is not available. Please set device to 'cpu'.")
        self.dataloader = dataloader

        # Validate source data dir
        if not exists(source_data_dir):
            raise ValueError(
                f"Source data directory {source_data_dir} does not exist.")
        self.source_data_dir = Path(source_data_dir)

        # Validate/create target directory for processed outputs
        if not exists(target_data_dir):
            os.makedirs(target_data_dir)
        self.target_data_dir = Path(target_data_dir)

        # HDF5 save location defaults to target directory if not provided
        if hdf5_dir:
            if not exists(hdf5_dir):
                os.makedirs(hdf5_dir)
            self.hdf5_dir = Path(hdf5_dir)
        else:
            self.hdf5_dir = self.target_data_dir

        self.hdf5_file_name = hdf5_file_name
        print(f"Using HDF5 directory: {self.hdf5_dir}")

        if not selected_timesteps:
            raise ValueError(
                "Selected timesteps must be provided and cannot be empty.")
        self.selected_timesteps = sorted(selected_timesteps, reverse=True)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.log_every = log_every
        self.end_batch_id = end_batch_id
        self.group_name = run_type

        self.start_file_idx = start_file_idx
        self.current_file_idx = None

        # Setup DiT-XL/2 model
        y_null = torch.tensor([self.num_classes] *
                              self.batch_size, device=self.device)
        self.sample_kwargs = sample_kwargs or {}
        self.sample_kwargs.update({
            'num_steps': num_steps,
            'cfg_scale': 1.0,
            'uc_cond': y_null,
            'cond_key': 'y'
        })

        # Load models
        self.first_stage = torch.compile(AutoencoderKL(
            ckpt_path=first_stage_ckpt).to(self.device), fullgraph=True)
        freeze(self.first_stage)
        self.first_stage.eval()

        net = DiT_models["DiT-XL/2"](
            input_size=self.input_size,
            num_classes=self.num_classes,
            learn_sigma=True,
            load_from_ckpt=second_stage_ckpt
        ).to(self.device)
        flow_model = FlowModel(net, schedule="linear").to(self.device)
        self.second_stage = torch.compile(flow_model, fullgraph=True)
        freeze(self.second_stage)
        self.second_stage.eval()

    @torch.no_grad()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        return self.first_stage.encode(x) if self.first_stage else x

    @torch.no_grad()
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        return self.first_stage.decode(z) if self.first_stage else z

    @torch.no_grad()
    def encode_second_stage(self, latent: torch.Tensor, y: Optional[torch.Tensor] = None,
                            return_intermediates: bool = True, sample_kwargs: Optional[Dict] = None) -> Tuple:
        return self.second_stage.encode(latent, y=y, return_intermediates=return_intermediates, **(sample_kwargs or {}))

    @torch.no_grad()
    def decode_second_stage(self, z: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.second_stage.generate(z, y=label, **self.sample_kwargs)

    @torch.no_grad()
    def process(self) -> None:
        print(f"Selected timesteps: {self.selected_timesteps}")
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > self.end_batch_id:
                break

            x = images.to(self.device).float()
            y = labels.to(self.device).long()

            latent = self.encode_first_stage(x)
            xt, intermediates = self.encode_second_stage(
                latent, y=y, return_intermediates=True, sample_kwargs=self.sample_kwargs)

            filtered_intermediates = {
                f"{t:.1f}": intermediates.get(f"{t:.1f}")
                for t in self.selected_timesteps
                if intermediates.get(f"{t:.1f}") is not None
            }

            if batch_idx % self.log_every == 0:
                print(
                    f"Batch {batch_idx}/{self.end_batch_id} - {self.group_name}")
                show_samples(filtered_intermediates, save_dir=str(
                    self.target_data_dir), prefix=f"{self.group_name}_samples_{batch_idx}")

            data_dict = {
                'image': x.detach().cpu(),
                'latent': xt.detach().cpu(),
                'label': y.detach().cpu(),
                'intermediate_steps': self.selected_timesteps,
                'intermediates': list(filtered_intermediates.values()),
            }

            # Save images, labels, and latents to the target directory
            self.save_files_to_data_dir(data_dict, group_name=self.group_name)
            torch.cuda.empty_cache()
            gc.collect()

        hdf5_filename = self.hdf5_file_name or f'imagenet256-testset-{datetime.datetime.now().strftime("T%H%M%S")}.hdf5'
        print(f"Save to hdf5 file {hdf5_filename}.")

        self.save_to_hdf5(filename=hdf5_filename, group_name=self.group_name)
        torch.cuda.empty_cache()
        gc.collect()

    """    Save to HDF5 (New Version)   """

    def save_to_hdf5(self,
                     group_name='train',
                     img_shape=(3, 256, 256),
                     latent_shape=(4, 32, 32),
                     label_shape=(1,),
                     filename='data.hdf5',
                     timesteps=None):

        file_path = self.hdf5_dir / filename
        print(f"[INFO] Saving data to HDF5 file: {file_path}")

        group_dir = Path(self.target_data_dir) / group_name
        images_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        print(f"[INFO] Group directory: {group_dir}")
        print(f"[INFO] Images directory: {images_dir}")
        print(f"[INFO] Labels directory: {labels_dir}")

        if timesteps is not None:
            timestep_names = {f"Latents_{t:.2f}" for t in timesteps}
            latent_timestep_dirs = [d for d in group_dir.glob(
                'Latents_*') if d.name in timestep_names]
        else:
            latent_timestep_dirs = sorted(group_dir.glob('Latents_*'))

        print(
            f"[INFO] Found latent timestep directories: {latent_timestep_dirs}")

        if not images_dir.exists() or not labels_dir.exists():
            print(
                f"[ERROR] Required directories (Images/Labels) do not exist in {group_dir}")
            return None

        with h5py.File(file_path, 'a') as h5_file:
            group = h5_file.require_group(group_name)

            # Check for or create image dataset
            if 'images' not in group:
                img_dset = group.create_dataset(
                    'images', shape=(0,) + img_shape, maxshape=(None,) + img_shape,
                    dtype=np.float32, chunks=(1,) + img_shape, compression="gzip"
                )
            else:
                img_dset = group['images']

            # Check for or create label dataset
            if 'labels' not in group:
                lbl_dset = group.create_dataset(
                    'labels', shape=(0,) + label_shape, maxshape=(None,) + label_shape,
                    dtype=np.int64, chunks=(1,) + label_shape, compression="gzip"
                )
            else:
                lbl_dset = group['labels']

            image_paths = sorted(images_dir.glob('*.png'),
                                 key=lambda p: int(p.stem))
            label_paths = sorted(labels_dir.glob('*.npy'),
                                 key=lambda p: int(p.stem))

            # Assert equal number of images and labels
            assert len(image_paths) == len(
                label_paths), "Number of images and labels do not match!"

            # Append images and labels
            for img_path, label_path in zip(image_paths, label_paths):
                # Load image
                with Image.open(img_path) as im:
                    img = np.asarray(im.convert('RGB'), dtype=np.float32)
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW

                # Load label
                label = np.load(label_path).astype(
                    np.int64).reshape(label_shape)

                # Resize datasets to append
                img_dset.resize(img_dset.shape[0] + 1, axis=0)
                lbl_dset.resize(lbl_dset.shape[0] + 1, axis=0)

                img_dset[-1] = img
                lbl_dset[-1] = label

            # Process latent timesteps
            for timestep_dir in latent_timestep_dirs:
                timestep = timestep_dir.name.split('_')[-1]
                latent_group_name = f'latents_{timestep}'

                print(
                    f"[INFO] Processing latent group: {latent_group_name} in {timestep_dir}")

                if latent_group_name not in group:
                    lat_dset = group.create_dataset(
                        latent_group_name, shape=(0,) + latent_shape, maxshape=(None,) + latent_shape,
                        dtype=np.float32, chunks=(1,) + latent_shape, compression="gzip"
                    )
                else:
                    lat_dset = group[latent_group_name]

                latent_paths = sorted(timestep_dir.glob('*.npy'))

                for latent_path in latent_paths:
                    latent = np.load(latent_path).astype(np.float32)
                    lat_dset.resize(lat_dset.shape[0] + 1, axis=0)
                    lat_dset[-1] = latent

        print(f"Data appended to {file_path}.")
        return file_path

    def ensure_chw_format(latent: torch.Tensor) -> torch.Tensor:
        """Ensure latent is in (C, H, W) format."""
        if latent.ndim == 3 and latent.shape[0] <= 4:
            # Likely already (C, H, W)
            return latent
        elif latent.ndim == 3 and latent.shape[-1] <= 4:
            # Likely (H, W, C)
            return latent.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")

    def get_last_index(self, img_dir: str) -> int:
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        if not image_files:
            return 0
        last_index = max(int(re.search(r'(\d+)\.png', f).group(1))
                         for f in image_files)
        return last_index

    def save_files_to_data_dir(self, data: Dict[str, Any], group_name: str = 'train') -> str:
        """
        Store images, labels, and latents in a structured directory format.
        Example dataset structure:
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
        group_dir = self.target_data_dir / group_name
        img_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        self.create_directory(group_dir)
        self.create_directory(img_dir)
        self.create_directory(labels_dir)

        # Increment index
        if self.current_file_idx is None:
            self.current_file_idx = self.start_file_idx
        else:
            self.current_file_idx = self.get_last_image_number(img_dir) + 1

        #######################################
        # Save images, labels, and latents
        start_index = self.current_file_idx
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
        #######################################

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
                img_data = self.ensure_chw_format(img_data)
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
                latent_data = self.ensure_chw_format(
                    latent_data).astype(np.float32)
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

    def get_last_image_number(self, directory: Path) -> int:
        """Get the last image number in the directory."""
        try:
            existing_files = [f for f in directory.glob('*.png')]
            numbers = []
            for f in existing_files:
                try:
                    numbers.append(int(f.stem))
                except ValueError:
                    continue  # skip non-numeric filenames
            return max(numbers) if numbers else 0
        except Exception as e:
            logging.error(
                f"Error getting last image number from {directory}: {e}")
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


# ----------- Example Usage -----------

def main():
    base_data_dir = "./dataset/train-samples"
    target_data_dir = "./dataset/processed/trainset-256/"
    hdf5_dir = "./dataset/processed/trainset-256/"

    start_file_idx = 479235  # Optional, defaults to 0

    # Optional, defaults to imagenet256-testset-T%H%M%S.hdf5
    hdf5_file_name = 'imagenet256-dataset-T000006.hdf5'

    batch_size = 32
    num_workers = 8
    selected_timesteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset = ClassIDImageDataset(base_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    processor = SampleProcessor(
        dataloader=dataloader,
        # original/raw data location
        source_data_dir=base_data_dir,
        # where to save processed samples
        target_data_dir=target_data_dir,
        # where to save hdf5 files, defaults to target_data_dir
        hdf5_dir=hdf5_dir,
        # Optional - HDF5 file name, defaults to imagenet256-testset-T%H%M%S.hdf5
        hdf5_file_name=hdf5_file_name,
        selected_timesteps=selected_timesteps,
        start_file_idx=start_file_idx,
        first_stage_ckpt='checkpoints/sd_ae.ckpt',
        second_stage_ckpt='checkpoints/SiT-XL-2-256x256.pt',
        input_size=32,  # image size wxh
        num_classes=1000,
        batch_size=batch_size,
        num_steps=100,
        log_every=1000,
        run_type='train'
    )

    processor.process()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python ...
