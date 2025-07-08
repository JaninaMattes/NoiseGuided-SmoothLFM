import os
import re
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Tuple
from pathlib import Path

import h5py
import webdataset as wds
import logging

import torch


from ldm.dataloader.dataloader.dataloader import HDF5DataModule





#############################################################
#                   HDF5 Data Handler (old)                 #
#############################################################
# - old version of the HDF5 data handler
# - used for saving and loading data in HDF5 format
class HDF5DataHandler:
    def __init__(self, file_path: str = 'data', generate=False):
        self.data_dir = file_path
        if generate:
            os.makedirs(self.data_dir, exist_ok=True)

    def save_intermediates(self, group: h5py.Group, data: Dict[str, Any]) -> None:
        if 'intermediates' and 'intermediate_steps' in data:
            intermediates = data['intermediates']
            intermediate_steps = data['intermediate_steps']

            for enc_img, t_step in zip(intermediates, intermediate_steps):
                t_step_float = t_step.item() if isinstance(t_step, torch.Tensor) else t_step
                enc_key = f'latefPathnt_{t_step_float:.1f}'
                self.create_or_append(
                    group, enc_key, enc_img.cpu().numpy(), dtype=np.float32)
        else:
            logging.warning("No intermediate data to save.")

    def create_or_append(self, group: h5py.Group, key: str, new_data: np.ndarray, dtype: np.dtype) -> None:
        if key not in group:
            maxshape = (None,) + new_data.shape[1:]
            group.create_dataset(key, data=new_data, compression="gzip",
                                 chunks=True, maxshape=maxshape, dtype=dtype)
        else:
            group[key].resize(
                (group[key].shape[0] + new_data.shape[0]), axis=0)
            group[key][-new_data.shape[0]:] = new_data

    def get_last_index(self, img_dir: str) -> int:
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        if not image_files:
            return 0
        last_index = max(int(re.search(r'(\d+)\.png', f).group(1))
                         for f in image_files)
        return last_index

    def save_to_numpy(self, data: Dict[str, Any], group_name: str = 'train') -> str:
        img_dir = os.path.join(self.data_dir, 'Images')
        latent_dir = os.path.join(self.data_dir, 'Latents')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)

        last_image_number = self.get_last_image_number(img_dir)
        start_index = last_image_number + 1

        if 'image' in data:
            images = data['image']
            for i, image in enumerate(images):
                image_filename = os.path.join(
                    img_dir, f'{start_index + i:05d}.png')
                plt.imsave(image_filename, image.transpose(
                    1, 2, 0))  # Transpose to (H, W, C)

        if 'latent' in data:
            latents = data['latent']
            for i, latent in enumerate(latents):
                latent_filename = os.path.join(
                    latent_dir, f'Latents_0.0', f'{start_index + i:05d}.npy')
                os.makedirs(os.path.dirname(latent_filename), exist_ok=True)
                np.save(latent_filename, latent)

        if 'intermediate_steps' in data and 'intermediates' in data:
            intermediate_steps = data['intermediate_steps']
            intermediates = data['intermediates']
            for step, intermediates_batch in zip(intermediate_steps, intermediates):
                for i, intermediate in enumerate(intermediates_batch):
                    latent_filename = os.path.join(
                        latent_dir, f'Latents_{step:.1f}', f'{start_index + i:05d}.npy')
                    os.makedirs(os.path.dirname(
                        latent_filename), exist_ok=True)
                    np.save(latent_filename, intermediate)

        logging.info(f"Data has been stored to {self.data_dir}.")
        return self.data_dir

    def save_to_hdf5(self, data: Dict[str, Any], group_name: str = 'train', filename='data.hdf5') -> str:
        file_path = os.path.join(self.data_dir, filename)
        with h5py.File(file_path, 'a') as h5_file:
            group = h5_file.require_group(group_name)

            if 'image' in data:
                self.create_or_append(
                    group, 'image', data['image'], dtype=np.float32)
            if 'latent' in data:
                self.create_or_append(
                    group, 'latent', data['latent'], dtype=np.float32)
            if 'image_recon' in data:
                self.create_or_append(
                    group, 'image_recon', data['image_recon'], dtype=np.float32)
            if 'label' in data:
                self.create_or_append(
                    group, 'label', data['label'], dtype=np.uint8)

            self.save_intermediates(group, data)

            logging.info(f"Data has been stored to {self.data_dir}.")

        return file_path

    def retrieve_keys_from_hdf5(self, group_name: str = 'train') -> List[str]:
        keys = []
        try:
            with h5py.File(self.data_dir, 'r') as h5_file:
                group = h5_file[group_name]
                keys = list(group.keys())
        except Exception as e:
            logging.error(f"Error retrieving keys from HDF5: {e}")
        return keys

    def load_data_by_keys(self, keys: List[str], group_name: str = 'train') -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        data = {}
        try:
            with h5py.File(self.data_dir, 'r') as h5_file:
                group = h5_file[group_name]

                for key in keys:
                    if key in group:
                        x = torch.tensor(group[key][:], dtype=torch.float32)
                        labels = torch.tensor(
                            group['label'][:], dtype=torch.uint8)
                        data[key] = (x, labels)
        except Exception as e:
            logging.error(f"Error loading data from HDF5: {e}")
        return data

    def load_all_data_from_hdf5(self, group_name: str = 'train') -> Dict[str, Any]:
        data = {}
        try:
            with h5py.File(self.data_dir, 'r') as h5_file:
                group = h5_file[group_name]

                for key in group.keys():
                    if key == 'label':
                        data[key] = torch.tensor(
                            group[key][:], dtype=torch.uint8)
                    else:
                        data[key] = torch.tensor(
                            group[key][:], dtype=torch.float32)

        except Exception as e:
            logging.error(f"Error loading all data from HDF5: {e}")
        return data

    def load_from_hdf5(
        self,
        base_dir: Path,
        batch_size: int = 16,
        timestep: int = 0.0
    ) -> wds.WebLoader:
        """
        Load data using WebLoader from the specified numpy directory structure.
        """
        loader = HDF5DataModule(
            base_dir=base_dir,
            batch_size=batch_size,
            timestep=timestep
        )

        loader.setup(stage='fit')

        return loader


if __name__ == "__main__":
    # Example usage
    file_path = '/data'
    data_handler = HDF5DataHandler(file_path)

    data = {
        'image': np.random.rand(8, 3, 32, 32),
        'latent': np.random.rand(8, 4, 32, 32),
        'image_recon': np.random.rand(8, 3, 32, 32),
        'label': np.random.randint(0, 10, 8),
        'intermediate_steps': [0.0, 0.5],
        'intermediates': [np.random.rand(8, 3, 32, 32), np.random.rand(8, 3, 32, 32)],
    }

    sample_dataset = [data for _ in range(10)]

    for i, data in enumerate(sample_dataset):
        # data_handler.save_to_hdf5(data)
        data_handler.save_to_numpy(data)
