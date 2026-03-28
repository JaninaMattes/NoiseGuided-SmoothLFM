# Code adapted from:
# - https://github.com/joh-schb/image-ldm/blob/main/ldm/dataloader.py
# Description: PyTorch Lightning DataModule for loading datasets from WebDataset, HDF5 files, and dummy datasets.
import os
import torch
import logging
import logging
import os
from pathlib import Path
import random
import sys
import h5py

from PIL import Image
import numpy as np

import torchvision
import webdataset as wds

from omegaconf import OmegaConf
from omegaconf import ListConfig

# Percentile normalization
from sklearn.preprocessing import QuantileTransformer

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
import pytorch_lightning as pl

from jutils import exists, default
from jutils import instantiate_from_config
from jutils import load_partial_from_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

# from ldm.tools.transform.transforms import CustomMultiRandSiTFlip, CustomRandHorizontalFlip
    




##################################################################
#                         WebDataset                             #
##################################################################



""" WebDataset """


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


def identity(x):
    return x




class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 tar_base,          # can be a list of paths or a single path
                 batch_size,
                 val_batch_size=None,
                 train=None,
                 validation=None,
                 test=None,
                 num_workers=4,
                 val_num_workers: int = None,
                 multinode=True,
                 remove_keys: list = None,          # list of keys to remove from the sample
                 ):
        super().__init__()
        if isinstance(tar_base, str):
            self.tar_base = tar_base
        elif isinstance(tar_base, ListConfig) or isinstance(tar_base, list):
            # check which tar_base exists
            for path in tar_base:
                if os.path.exists(path):
                    self.tar_base = path
                    break
            else:
                raise FileNotFoundError("Could not find a valid tarbase.")
        else:
            raise ValueError(f'Invalid tar_base type {type(tar_base)}')
        print(f'[WebDataModuleFromConfig] Setting tar base to {self.tar_base}')
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def make_loader(self, dataset_config, train=True):
        image_transforms = []
        lambda_fn = lambda x: x * 2. - 1.   # normalize to [-1, 1]
        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda_fn)])
        if 'image_transforms' in dataset_config:
            image_transforms.extend([instantiate_from_config(tt) for tt in dataset_config.image_transforms])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        # this is crucial to set correct image key to get the transofrms applied correctly
        img_key = dataset_config.get('image_key', 'image.png')
        transform_dict.update({img_key: image_transforms})

        if 'dataset_transforms' in dataset_config:
            dataset_transforms = instantiate_from_config(dataset_config['dataset_transforms'])
        else:
            dataset_transforms = None

        if 'postprocess' in dataset_config:
            postprocess = instantiate_from_config(dataset_config['postprocess'])
        else:
            postprocess = None

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if isinstance(dataset_config.shards, str):
            tars = os.path.join(self.tar_base, dataset_config.shards)
        elif isinstance(dataset_config.shards, list) or isinstance(dataset_config.shards, ListConfig):
            # decompose into lists of shards
            # Turn train-{000000..000002}.tar into ['train-000000.tar', 'train-000001.tar', 'train-000002.tar']
            tars = []
            for shard in dataset_config.shards:
                # Assume that the shard starts from 000000
                if '{' in shard:
                    start, end = shard.split('..')
                    start = start.split('{')[-1]
                    end = end.split('}')[0]
                    start = int(start)
                    end = int(end)
                    tars.extend([shard.replace(f'{{{start:06d}..{end:06d}}}', f'{i:06d}') for i in range(start, end+1)])
                else:
                    tars.append(shard)
            tars = [os.path.join(self.tar_base, t) for t in tars]
            # random shuffle the shards
            if shardshuffle:
                np.random.shuffle(tars)
        else:
            raise ValueError(f'Invalid shards type {type(dataset_config.shards)}')

        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'[WebDataModuleFromConfig] Loading {len(dset.pipeline[0].urls)} shards.')

        dset = (dset
                .decode('rgb', handler=wds.warn_and_continue)
                .map(self.filter_out_keys, handler=wds.warn_and_continue)
                .map_dict(**transform_dict, handler=wds.warn_and_continue)
                )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get('rename', None)
        if renaming is not None:
            dset = dset.rename(**renaming)

        if dataset_transforms is not None:
            dset = dset.map(dataset_transforms)

        if postprocess is not None:
            dset = dset.map(postprocess)
        
        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw)

        return loader

    def filter_out_keys(self, sample):
        for key in self.rm_keys:
            sample.pop(key, None)
        return sample
    
    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)



class DummyDataset(Dataset):
    def __init__(self, num_samples=500000, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.keys_shapes = {k: v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            key: (
                torch.randn(*shape) if len(shape) > 1
                else torch.randint(0, 10, (1,)).squeeze()       # e.g. class labels
            )
            for key, shape in self.keys_shapes.items()
        }
        
        
        

class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__()
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = instantiate_from_config(transform)

        if target_transform is not None:
            target_transform = instantiate_from_config(target_transform)
        self.dataset = torchvision.datasets.CIFAR10(
            root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return {"image": img, "label": target}
    
    



""" Helpers """
class MomentsPreprocessor:
    def __init__(self, moments_key="moments.npy", out_key="latent", scale: float = 0.18215, shift: float = 0.0):
        self.moments_key = moments_key
        self.out_key = out_key
        self.scale = scale
        self.shift = shift

    def __call__(self, sample):
        """
        Helper function for ImageNet first stage sampling using moments.
        https://github.com/joh-schb/jutils/blob/8440e65b6296897ec23f0c1f13199ca0e1be92e9/jutils/nn/kl_autoencoder.py#L45
        """
        moments = torch.tensor(sample[self.moments_key])

        mean, logvar = torch.chunk(moments, 2, dim=0)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)

        latent = mean + std * torch.randn(mean.shape).to(device=moments.device)
        latent = (latent + self.shift) * self.scale
        sample[self.out_key] = latent

        del sample[self.moments_key]

        return sample





##################################################################
#                           Normal Dataset                       #
##################################################################




""" Normal Dataset """
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 val_batch_size: int = None,
                 timestep: float = 0.0,
                 train: dict = None,
                 validation: dict = None,
                 test: dict = None,
                 shuffle_validation: bool = False,
                 num_workers: int = 0,
                 drop_last: bool = False,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.timestep = timestep
        # -------------------------
        # Dataset loading based on config
        self.train = train
        self.validation = validation
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.shuffle_validation = shuffle_validation
        self.drop_last = drop_last

        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"], 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=True,
            drop_last=self.drop_last
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["validation"], 
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,  
            shuffle=self.shuffle_validation,
            drop_last=self.drop_last
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"], 
            batch_size=self.val_batch_size,
            num_workers=self.num_workers, 
            shuffle=self.shuffle_validation,
            drop_last=self.drop_last
        )

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)





##################################################################
#                  Custom HDF5-based Datasets                    #
##################################################################


""" Dataset from HDF5 files """
class HDF5LatentIterableDataset(IterableDataset):
    def __init__(self, base_dir: str, group_name: str, timestep: float, start: int, end: int, transform=None):
        """
        HDF5-based PyTorch Dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            group_name (str): 'train', 'validation', or 'test'.
            timestep (float): The timestep of latents to load (e.g., 0.0, 0.25).
            transform (callable, optional): Transformations for images.
            lazy_loading (bool): Load data on-the-fly or pre-load into memory.
        """
        self.base_dir = base_dir
        self.group_name = group_name
        self.timestep = timestep
        self.transform = transform
        
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        
    def __iter__(self):
        for idx in range(self.start, self.end):
            with h5py.File(self.base_dir, 'r', libver='latest', swmr=True) as h5_file:
                image = torch.tensor(h5_file[self.group_name]['images'][idx], dtype=torch.float32)
                label = torch.tensor(h5_file[self.group_name]['labels'][idx], dtype=torch.long)
                if self.timestep <= 1.0:
                    latent = torch.tensor(h5_file[self.group_name]['latents'][idx], dtype=torch.float32)
                else:
                    latent = image

                yield {'image': image, 'label': label, 'latent': latent}

class HDF5MultiLatentsDataset(Dataset):
    def __init__(
        self, 
        base_dir: str, 
        group_name: str='train', 
        source_timestep: float = 0.0,
        target_timestep: float = 1.0,
        transform=None, 
        norm_transform=None, 
        lazy_loading=True
    ):
        """
        HDF5-based PyTorch Dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            group_name (str): 'train', 'validation', or 'test'.
            timestep (float): The timestep of latents to load (e.g., 0.0, 0.25).
            transform (callable, optional): Transformations for images.
            lazy_loading (bool): Load data on-the-fly or pre-load into memory.
        
        Returns:
            dict: A dictionary containing the image, latent, and label.
        """
        self.base_dir = base_dir
        self.group_name = group_name
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        self.transform = transform
        self.norm_transform = norm_transform
        self.lazy_loading = lazy_loading
        self.current_index = 0                      # Track current index
        self.dataset = None

        with h5py.File(self.base_dir, 'r') as h5_file:
            self.dataset_size = h5_file[group_name]['images'].shape[0]  # Get total no samples

            # Find available keys
            self.available_latents = [
                key for key in h5_file[group_name].keys() if 'latents' in key
            ]

            """ Load latent at specific timesteps"""
            if self.source_timestep <= 1.0:
                # Check timestep availability
                latent_key = f'latents_{self.source_timestep:.2f}'
                if latent_key not in self.available_latents:
                    raise ValueError(
                        f"Latent timestep {self.source_timestep} not found in the dataset. Available latents: {self.available_latents}")
                self.source_latent_key = latent_key
            else:
                self.source_latent_key = 'images'

            if self.target_timestep <= 1.0:
                # Check timestep availability
                latent_key = f'latents_{self.target_timestep:.2f}'
                if latent_key not in self.available_latents:
                    raise ValueError(
                        f"Latent timestep {self.target_timestep} not found in the dataset. Available latents: {self.available_latents}")
                self.target_latent_key = latent_key
            else:
                self.target_latent_key = 'images'
                

            # --- Eager loading ---
            # Load entire dataset into memory            
            # Works only with small dataset! 
            if not self.lazy_loading:
                print(f"[HDF5LatentsDataset] Loading dataset into memory...")
                self.labels = h5_file[group_name]['labels'][:]
                self.images = h5_file[group_name]['images'][:]
                self.source_latents = h5_file[group_name][self.source_latent_key][:]
                self.target_latents = h5_file[group_name][self.target_latent_key][:]
        
        
        print(f"[HDF5LatentsDataset] Dataset size: {self.dataset_size}")     
                
    def __len__(self):
        """ Get dataset size. """
        return self.dataset_size
    
    
    def __getitem__(self, idx):    
        """ Get sample from HDF5 file. """
        self.current_index = idx  # Store current index for resuming
                
        if not self.lazy_loading:
            # --- Eager loading ---
            image = self.images[idx] 
            label = self.labels[idx]
            latent = self.latents[idx]
            
        else:
            # --- Lazy loading ---
            self.dataset = h5py.File(self.base_dir, 'r')[self.group_name] \
                if self.dataset is None else self.dataset
                
            image = self.dataset['images'][idx]
            label = self.dataset['labels'][idx]
            source_latent = self.dataset[self.source_latent_key][idx]
            target_latent = self.dataset[self.target_latent_key][idx]
    
        image = torch.from_numpy(image).to(torch.float32)
        source_latent = torch.from_numpy(source_latent).to(torch.float32)
        target_latent = torch.from_numpy(target_latent).to(torch.float32)
        label = torch.tensor(label).to(torch.long)

        # Normalize image
        if self.norm_transform:
            image = self.norm_transform(image)

        return {
            'image': image,
            f'latents_{self.source_timestep:.2f}': source_latent,  # assuming latent is source_latent
            f'latents_{self.target_timestep:.2f}': target_latent,  # or replace with actual target_latent if different
            'label': label
        }

    def state_dict(self):
        """ Save dataset state (current iteration). """
        return {'current_index': self.current_index}

    def load_state_dict(self, state_dict):
        """ Restore dataset state. """
        self.current_index = state_dict.get('current_index', 0)
        
        

class HDF5MultiDataModule(pl.LightningDataModule):
    def __init__(self,
                 base_dir: str,                     # Directory containing the numpy files
                 batch_size: int,
                 val_batch_size: int = None,
                 train: bool = True,
                 validation: bool = True,
                 test: bool = False,
                 num_workers: int = 4,
                 val_num_workers: int = None,
                 source_timestep: float = 0.0,      # Source sample
                 target_timestep: float = 0.5,      # Target sample
                 pin_memory: bool = True,           # Optimize GPU memory
                 multinode: bool = True,            # Multi-node training
                 prefetch_factor: int = 2,          # Prefetch data
                 remove_keys: list = None,          # Keys to remove from sample
                 drop_last:bool = False,            # Drop last batch
                 lazy_loading:bool = True,         # Lazy loading
                 transform=None,                    # Transformation pipeline
                 norm_transform=None                # Normalization pipeline
                ):
        super().__init__()
        self.base_dir = os.path.abspath(base_dir) if isinstance(base_dir, str) else self.resolve_base_dir(base_dir)
        print(f'[HDF5DataModule] Setting base directory to {self.base_dir}')
        self.batch_size = batch_size
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        
        # -------------------------
        # Dataset loading based on config
        self.train = train
        self.validation = validation
        self.test = test
        self.norm_transform = norm_transform
        self.transform = transform
        
        # -------------------------
        self.multinode = multinode
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.lazy_loading = lazy_loading
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []


    def setup(self, stage=None):
        """Set up datasets for different stages. Called from all processes."""   
    
        target_transform = None     # TOOD: Needs Fix
        # Normalize [0, 255] to [-1, 1]
        mean = (127.0, 127.0, 127.0)
        std = (127.0, 127.0, 127.0)
        img_transform = transforms.Compose([
            torchvision.transforms.Normalize(mean, std),
        ])
        # Setup datasets
        if stage == 'fit':
            if self.train:
                self.train_dataset = HDF5MultiLatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='train', 
                    source_timestep=self.source_timestep, 
                    target_timestep=self.target_timestep,
                    transform=target_transform,
                    norm_transform=img_transform,
                    lazy_loading=self.lazy_loading
                )
                
            if self.validation:
                self.val_dataset = HDF5MultiLatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='validation', 
                    source_timestep=self.source_timestep, 
                    target_timestep=self.target_timestep,
                    transform=None,
                    norm_transform=img_transform,
                    lazy_loading=self.lazy_loading
                )
        
        if stage == 'test' and self.test:
            self.test_dataset = HDF5MultiLatentsDataset(
                base_dir=self.base_dir, 
                group_name='test', 
                source_timestep=self.source_timestep, 
                target_timestep=self.target_timestep,
                transform=None,
                norm_transform=None,
                lazy_loading=self.lazy_loading
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Did you call `setup(stage='fit')`?")
        use_multiprocessing = self.num_workers > 0
        train_data = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return train_data

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Did you call `setup(stage='fit')`?")
        use_multiprocessing = self.num_workers > 0
        val_data = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return val_data

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise  ValueError("Test dataset is not initialized. Did you call `setup(stage='test')`?")
        use_multiprocessing = self.num_workers > 0
        test_data = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return test_data


    def set_timestep(self, new_timestep: float):
        """ Dynamically update the timestep for loading latents. """
        self.timestep = new_timestep

        # Reinitialize datasets with the new timestep
        if self.train_dataset:
            self.train_dataset.timestep = self.timestep
        if self.val_dataset:
            self.val_dataset.timestep = self.timestep
        if self.test_dataset:
            self.test_dataset.timestep = self.timestep


    def resolve_base_dir(self, base_dir_list):
        for path in base_dir_list:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        raise FileNotFoundError("Could not find a valid base directory.")

                             
                                                   
                                   
# ------------------------------------------------------------------------------
class HDF5LatentsDataset(Dataset):
    def __init__(
        self, 
        base_dir: str, 
        group_name: str='train', 
        timestep: float=0.0, 
        transform=None, 
        norm_transform=None, 
        lazy_loading=True
    ):
        """
        HDF5-based PyTorch Dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            group_name (str): 'train', 'validation', or 'test'.
            timestep (float): The timestep of latents to load (e.g., 0.0, 0.25).
            transform (callable, optional): Transformations for images.
            lazy_loading (bool): Load data on-the-fly or pre-load into memory.
        
        Returns:
            dict: A dictionary containing the image, latent, and label.
        """
        self.base_dir = base_dir
        self.group_name = group_name
        self.timestep = timestep
        self.transform = transform
        self.norm_transform = norm_transform
        self.lazy_loading = lazy_loading
        self.current_index = 0                      # Track current index
        self.dataset = None

        with h5py.File(self.base_dir, 'r') as h5_file:
            self.dataset_size = h5_file[group_name]['images'].shape[0]  # Get total no samples

            # Find available keys
            self.available_latents = [
                key for key in h5_file[group_name].keys() if 'latents' in key
            ]

            """ Load latent at specific timesteps"""
            if self.timestep <= 1.0:
                # Check timestep availability
                latent_key = f'latents_{self.timestep:.2f}'
                if latent_key not in self.available_latents:
                    raise ValueError(
                        f"Latent timestep {self.timestep} not found in the dataset. Available latents: {self.available_latents}")
                self.latent_key = latent_key
            else:
                self.latent_key = 'images'


            # --- Eager loading ---
            # Load entire dataset into memory            
            # Works only with small dataset! 
            if not self.lazy_loading:
                print(f"[HDF5LatentsDataset] Loading dataset into memory...")
                self.labels = h5_file[group_name]['labels'][:]
                self.images = h5_file[group_name]['images'][:]
                self.latents = h5_file[group_name][self.latent_key][:]
        
        
        print(f"[HDF5LatentsDataset] Dataset size: {self.dataset_size}")     
                
    def __len__(self):
        """ Get dataset size. """
        return self.dataset_size
    
    
    def __getitem__(self, idx):    
        """ Get sample from HDF5 file. """
        self.current_index = idx  # Store current index for resuming
                
        if not self.lazy_loading:
            # --- Eager loading ---
            image = self.images[idx] 
            label = self.labels[idx]
            latent = self.latents[idx]
            
        else:
            # --- Lazy loading ---
            self.dataset = h5py.File(self.base_dir, 'r')[self.group_name] \
                if self.dataset is None else self.dataset
                
            image = self.dataset['images'][idx]
            label = self.dataset['labels'][idx]
            latent = self.dataset[self.latent_key][idx]
    
        image = torch.from_numpy(image).to(torch.float32)
        latent = torch.from_numpy(latent).to(torch.float32)
        label = torch.tensor(label).to(torch.long)          
        
        # Normalize image
        if self.norm_transform:
            image = self.norm_transform(image)
            
        return dict(image=image, latent=latent, label=label)
    

    def state_dict(self):
        """ Save dataset state (current iteration). """
        return {'current_index': self.current_index}

    def load_state_dict(self, state_dict):
        """ Restore dataset state. """
        self.current_index = state_dict.get('current_index', 0)
        
        
        

""" Pytorch Lightning DataModule from HDF5 Files """
class HDF5DataModule(pl.LightningDataModule):
    def __init__(self,
                 base_dir: str,                     # Directory containing the numpy files
                 batch_size: int,
                 val_batch_size: int = None,
                 train: bool = True,
                 validation: bool = True,
                 test: bool = False,
                 num_workers: int = 4,
                 val_num_workers: int = None,
                 timestep: float = 0.0,             # Default timestep
                 pin_memory: bool = True,           # Optimize GPU memory
                 multinode: bool = True,            # Multi-node training
                 prefetch_factor: int = 2,          # Prefetch data
                 remove_keys: list = None,          # Keys to remove from sample
                 drop_last:bool = False,            # Drop last batch
                 lazy_loading:bool = True,         # Lazy loading
                 transform=None,                    # Transformation pipeline
                 norm_transform=None                # Normalization pipeline
                ):
        super().__init__()
        self.base_dir = os.path.abspath(base_dir) if isinstance(base_dir, str) else self.resolve_base_dir(base_dir)
        print(f'[HDF5DataModule] Setting base directory to {self.base_dir}')
        self.batch_size = batch_size
        self.timestep = timestep
        # -------------------------
        # Dataset loading based on config
        self.train = train
        self.validation = validation
        self.test = test
        self.norm_transform = norm_transform
        self.transform = transform
        
        # -------------------------
        self.multinode = multinode
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.lazy_loading = lazy_loading
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def setup(self, stage=None):
        """Set up datasets for different stages. Called from all processes."""   
    
        target_transform = None     # TOOD: Needs Fix
        # Normalize [0, 255] to [-1, 1]
        mean = (127.0, 127.0, 127.0)
        std = (127.0, 127.0, 127.0)
        img_transform = transforms.Compose([
            torchvision.transforms.Normalize(mean, std),
        ])
        # Setup datasets
        if stage == 'fit':
            if self.train:
                self.train_dataset = HDF5LatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='train', 
                    timestep=self.timestep, 
                    transform=target_transform,
                    norm_transform=img_transform,
                    lazy_loading=self.lazy_loading
                )
                
            if self.validation:
                self.val_dataset = HDF5LatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='validation', 
                    timestep=self.timestep, 
                    transform=None,
                    norm_transform=img_transform,
                    lazy_loading=self.lazy_loading
                )
        
        if stage == 'test' and self.test:
            self.test_dataset = HDF5LatentsDataset(
                base_dir=self.base_dir, 
                group_name='test', 
                timestep=self.timestep, 
                transform=None,
                norm_transform=None,
                lazy_loading=self.lazy_loading
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Did you call `setup(stage='fit')`?")
        use_multiprocessing = self.num_workers > 0
        train_data = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return train_data

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Did you call `setup(stage='fit')`?")
        use_multiprocessing = self.num_workers > 0
        val_data = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return val_data

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise  ValueError("Test dataset is not initialized. Did you call `setup(stage='test')`?")
        use_multiprocessing = self.num_workers > 0
        test_data = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if use_multiprocessing else 2,
            persistent_workers=use_multiprocessing,
            drop_last=self.drop_last
        )
        
        return test_data


    def set_timestep(self, new_timestep: float):
        """ Dynamically update the timestep for loading latents. """
        self.timestep = new_timestep

        # Reinitialize datasets with the new timestep
        if self.train_dataset:
            self.train_dataset.timestep = self.timestep
        if self.val_dataset:
            self.val_dataset.timestep = self.timestep
        if self.test_dataset:
            self.test_dataset.timestep = self.timestep


    def resolve_base_dir(self, base_dir_list):
        for path in base_dir_list:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        raise FileNotFoundError("Could not find a valid base directory.")

                            




#################################################################################
#                        Simple Dataset from NPY Files                          #
#################################################################################



""" WebDataset from Numpy Arrays """
class NumpyLatentsDataset(Dataset):
    def __init__(self, base_dir: str, group_name='train', timestep: float=0.0, q_normalize=False, transform=None):
        """
        Args:
            base_dir (str): Base directory containing `Images`, `Labels`, and `Latents_<timestep>` folders.
            group_name (str): Specifies whether to load `train`, `validation`, or `test`.
            timestep (float): Specific timestep to load latents.
            transform (callable, optional): Transformations to apply to images.
        """
        self.base_dir = base_dir
        self.group_name = group_name
        self.timestep = timestep
        self.transform = transform
        self.q_normalize = q_normalize
        
        
        # Directories for Images, Labels, and Latents
        self.img_dir = Path(self.base_dir) / group_name / 'Images'
        self.label_dir = Path(self.base_dir) / group_name / 'Labels'
        
        if self.timestep <= 1.0:
            self.latent_dir = Path(self.base_dir) / group_name / f'Latents_{self.timestep:.2f}'
        else:
            # exception for pixel-space training
            self.latent_dir = self.img_dir

        # Preload file mappings
        self.img_paths, self.label_paths, self.latent_paths = self._preload_file_paths()


    def _preload_file_paths(self):
        """
        Preload and map file paths for Images, Labels, and Latents.
        Assumes all files share the same base filename (e.g., 00001.png, 00001.npy).
        """
        # Ensure directories exist
        for dir_path in [self.img_dir, self.label_dir, self.latent_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        img_files = list(self.img_dir.glob("*.png"))
        label_files = list(self.label_dir.glob("*.npy"))
        
        if self.timestep <= 1.0:
            latent_files = list(self.latent_dir.glob("*.npy"))
        else:
            latent_files = list(self.latent_dir.glob("*.png"))

        img_files = sorted([str(f) for f in img_files])
        label_files = sorted([str(f) for f in label_files])
        latent_files = sorted([str(f) for f in latent_files])

        return img_files, label_files, latent_files
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        latent_path = self.latent_paths[idx]
        image = self._load_image(image_path)
        label = self._load_label(label_path)
        
        if self.timestep <= 1.0:
            # Latent domain (no normalization)
            latent = self._load_latent(latent_path)
        else:
            # Pixel domain  [0, 255] → [-1, 1]
            latent = image
            if latent.max() > 1.0:
                latent = (latent / 127.5) - 1.0 
        
        if self.q_normalize:
            latent = self._normalize_quantiles(latent)
        
        sample = {'image': image, 'latent': latent, 'label': label}
        
        # Augmentations
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _find_img_paths(self):
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        paths = list(Path(self.img_dir).glob('*.png'))
        if len(paths) == 0:
            logging.warning(f"No images found in directory: {self.img_dir}")
        return paths


    def _load_image(self, image_path) -> torch.Tensor:
        try:
            # Load image and convert to RGB if RGBA
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # Convert to numpy
                img = np.asarray(img).astype(np.float32)
                # if (H, W, C) -->  (C, H, W)
                if img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).to(torch.float32)
                return img
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            raise

    def _load_latent(self, latent_path) -> torch.Tensor:
        try:
            # Load latent data
            latent = np.load(latent_path, allow_pickle=True)
            # if (H, W, C) -->  (C, H, W)
            if latent.shape[-1] == 4:
                latent = np.transpose(latent, (2, 0, 1))            # Convert (H, W, C) → (C, H, W)
            latent = torch.from_numpy(latent).to(torch.float32)
            return latent
        except Exception as e:
            logging.error(f"Error loading latent file {latent_path}: {e}")
            raise

    def _load_label(self, label_file) -> torch.Tensor:
        try:
            # Load label data
            label = np.load(label_file)
            label = torch.from_numpy(label).long()
            return label
        except Exception as e:
            logging.error(f"Error loading label file {label_file}: {e}")
            raise
    
    def _normalize_quantiles(self, tensor):
        """
        Normalize tensor to [0, 1] based on quantiles.
        Args:
            tensor (torch.Tensor): The input tensor to normalize.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        tensor_flat = tensor.flatten().reshape(-1, 1).detach().cpu().numpy()
        quantile = QuantileTransformer(output_distribution='normal')
        normed_data = quantile.fit_transform(tensor_flat)
        tensor = torch.tensor(normed_data).reshape(tensor.shape).to(tensor.device)
        return tensor






class WebDataModuleFromNumpyConfig(pl.LightningDataModule):
    def __init__(self,
                 base_dir: str,                     # Directory containing the numpy files
                 batch_size: int,
                 val_batch_size: int = None,
                 train: bool = True,
                 validation: bool = True,
                 test: bool = False,
                 num_workers: int = 4,
                 val_num_workers: int = None,
                 timestep: float = 0.0,             # Default timestep
                 pin_memory: bool = True,           # Optimize GPU memory
                 multinode: bool = True,            # Multi-node training
                 prefetch_factor: int = 2,          # Prefetch data
                 remove_keys: list = None,          # Keys to remove from sample
                 drop_last:bool = False             # Drop last batch
                ):
        super().__init__()
        self.base_dir = base_dir if isinstance(base_dir, str) else self.resolve_base_dir(base_dir)
        print(f'[WebDataModuleFromNumpyConfig] Setting base directory to {self.base_dir}')

        self.batch_size = batch_size
        self.timestep = timestep
        # -------------------------
        # Dataset loading based on config
        self.train = train
        self.validation = validation
        self.test = test
        # -------------------------
        self.multinode = multinode
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys or []
            
            
    def setup(self, stage=None):
        """Set up datasets for different stages. Called from all processes."""
        self.target_transform = None    # TODO: Needs Fix
        
        # transforms.Compose([
        #     CustomRandSiTFlip(prob=0.5)
        #     # Optional others
        # ])
        
        if stage == 'fit':
            self.train_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='train', 
                timestep=self.timestep, 
                transform=self.target_transform
            )
            self.val_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='validation', 
                timestep=self.timestep, 
                transform=None
            )
        if stage == 'test':
            self.test_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='test', 
                timestep=self.timestep, 
                transform=None
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        if self.train_dataset is None:
            self.train_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='train', 
                timestep=self.timestep, 
                transform=self.target_transform
            )
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            self.val_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='validation', 
                timestep=self.timestep, 
                transform=None
            )
        
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            self.test_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='test', 
                timestep=self.timestep, 
                transform=None
            )
        
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def set_timestep(self, new_timestep: float):
        """ Dynamically update the timestep for loading latents. """
        self.timestep = new_timestep
    
        # reload data 
        if self.train_dataset is not None:
            self.train_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='train', 
                timestep=self.timestep, 
                transform=self.target_transform
            )
        
        if self.val_dataset is not None:
            self.val_dataset = NumpyLatentsDataset(
                base_dir=self.base_dir, 
                group_name='validation', 
                timestep=self.timestep, 
                transform=None
            )
        
        
            
    def resolve_base_dir(self, base_dir_list):
        for path in base_dir_list:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("Could not find a valid base directory.")
                                        





class WebDataModuleFromNumpy(pl.LightningDataModule):
    def __init__(self,
                 base_dir,                          # Directory containing the numpy files
                 batch_size: int,
                 val_batch_size: int = None,
                 train: dict = None,
                 validation: dict = None,
                 test: dict = None,
                 num_workers: int = 0,
                 multinode:bool =True,
                 remove_keys: list = None,          # List of keys to remove from the sample
                 drop_last: bool = False            # Drop last batch
        ):
        super().__init__()
        self.base_dir = base_dir
        print(f'[WebDataModuleFromNumpy] Setting base directory to {self.base_dir}')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.rm_keys = remove_keys if remove_keys is not None else []
        self.drop_last = drop_last

    def make_loader(self, group_name, timestep:float=0.0, train:bool=True) -> DataLoader:
        # Transformation pipeline
        transform = None        # TODO: Needs Fix
        
        # transforms.Compose([ 
                                        
        #         # transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        #         # transforms.ToTensor(),
        #         # transforms.Lambda(lambda x: x * 2. - 1.)  # Normalize to [-1, 1]
        #     ])

        dataset = NumpyLatentsDataset(
            base_dir=self.base_dir,
            timestep=timestep,
            group_name=group_name,
            transform=transform # Add transforms
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size if train else self.val_batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 1 else None, # Overlap data loading with model training
            drop_last=self.drop_last
        )

    def train_dataloader(self, timestep: float = 0.0):
        return self.make_loader('train', timestep, train=True)

    def val_dataloader(self, timestep: float = 0.0):
        return self.make_loader('validation', timestep, train=False)

    def test_dataloader(self, timestep: float = 0.0):
        return self.make_loader('test', timestep, train=False)





class WebDataModuleFromHDF5(pl.LightningDataModule):
    def __init__(self,
                 base_dir: str,                     # Directory containing the numpy files
                 batch_size: int,
                 val_batch_size: int = None,
                 train: bool = True,
                 validation: bool = True,
                 test: bool = False,
                 num_workers: int = 4,
                 val_num_workers: int = None,
                 timestep: float = 0.0,             # Default timestep
                 pin_memory: bool = True,           # Optimize GPU memory
                 multinode: bool = True,            # Multi-node training
                 prefetch_factor: int = 2,          # Prefetch data
                 remove_keys: list = None,          # Keys to remove from sample
                 drop_last:bool = False,            # Drop last batch
                 lazy_loading:bool = False          # Lazy loading
                ):
        super().__init__()
        self.base_dir = base_dir if isinstance(base_dir, str) else self.resolve_base_dir(base_dir)
        print(f'[WebDataModuleFromNumpyConfig] Setting base directory to {self.base_dir}')

        self.batch_size = batch_size
        self.timestep = timestep
        # -------------------------
        # Dataset loading based on config
        self.train = train
        self.validation = validation
        self.test = test
        # -------------------------
        self.multinode = multinode
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.lazy_loading = lazy_loading
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys or []

    def setup(self, stage=None):
        """Set up datasets for different stages. Called from all processes."""
        transform = transforms.Compose([
            CustomRandHorizontalFlip(prob=0.5)
            # Optional others
        ])
        
        if stage == 'fit':
            if self.train:
                self.train_dataset = HDF5LatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='train', 
                    timestep=self.timestep, 
                    transform=transform,
                    lazy_loading=self.lazy_loading
                )
            if self.validation:
                self.val_dataset = HDF5LatentsDataset(
                    base_dir=self.base_dir, 
                    group_name='validation', 
                    timestep=self.timestep, 
                    transform=None,
                    lazy_loading=self.lazy_loading
                )
        
        if stage == 'test' and self.test:
            self.test_dataset = HDF5LatentsDataset(
                base_dir=self.base_dir, 
                group_name='test', 
                timestep=self.timestep, 
                transform=None,
                lazy_loading=self.lazy_loading
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Did you call `setup(stage='fit')`?")
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Did you call `setup(stage='fit')`?")
        
        return DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise  ValueError("Test dataset is not initialized. Did you call `setup(stage='test')`?")

        return DataLoader(
            self.test_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last
        )

    def set_timestep(self, new_timestep: float):
        """ Dynamically update the timestep for loading latents. """
        self.timestep = new_timestep

        # Reinitialize datasets with the new timestep
        if self.train_dataset:
            self.train_dataset.timestep = self.timestep
        if self.val_dataset:
            self.val_dataset.timestep = self.timestep
        if self.test_dataset:
            self.test_dataset.timestep = self.timestep


    def resolve_base_dir(self, base_dir_list):
        for path in base_dir_list:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("Could not find a valid base directory.")
                                        






#################################################################################
#                        Dummy Datasets for Testing                             #
#################################################################################




""" Dummy Dataset for ImageNet256 """

class DummyDataset(Dataset):
    def __init__(self, num_samples=500000, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.keys_shapes = {k: v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            key: (
                torch.randn(*shape) if len(shape) > 1
                else torch.randint(0, 10, (1,)).squeeze()       # e.g. class labels
            )
            for key, shape in self.keys_shapes.items()
        }



class LatentDummyDataset(Dataset):
    def __init__(self, classes,  timestep=0, num_samples=2000, transform=None, **kwargs):
        self.classes = classes
        self.timestep = timestep
        self.num_samples = num_samples
        self.transform = transform  # Optional transformation
        self.keys_shapes = {k: v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            key: (
                torch.randn(*shape) if len(shape) > 1
                else torch.randint(0, 10, (1,)).squeeze()       # e.g. class labels
            )
            for key, shape in self.keys_shapes.items()
        }



class ImageNet256MultiDummyDataset(Dataset):
    def __init__(self, classes, target_timestep=0.0, source_timestep=1.0, num_samples=2000, transform=None, **kwargs):
        """
        Args:
            classes (list): List of class labels.
            timestep (list): List of timestep (e.g., [0.00, 0.50, 1.00]).
            num_samples (int): Number of samples in the dataset.
            transform (callable, optional): Optional transform to apply to the image and latents.
            kwargs: Additional keys and their corresponding tensor shapes.
        """
        self.classes = classes
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        self.num_samples = num_samples
        self.keys_shapes = {k: tuple(map(int, v)) for k, v in kwargs.items()}
        self.transform = transform

    def __len__(self):
        return self.num_samples  # 500000 

    def __getitem__(self, idx):
        sample = {}

        # Generate image tensor (dummy data)
        sample["image"] = torch.randn(3, 256, 256, dtype=torch.float32)

        # Generate latents for each timestep
        sample[f"latents_{self.source_timestep:.2f}"] = torch.randn(4, 32, 32, dtype=torch.float32)
        sample[f"latents_{self.target_timestep:.2f}"] = torch.randn(4, 32, 32, dtype=torch.float32)

        # Generate label
        sample["label"] = torch.tensor(random.choice(self.classes), dtype=torch.long)

        # Add any additional keys with specified shapes
        for key, shape in self.keys_shapes.items():
            sample[key] = (
                torch.randn(*(int(dim) for dim in shape)) if len(shape) > 1
                else torch.tensor(random.choice(self.classes), dtype=torch.long)
            )

        # Apply transformation if provided
        if self.transform:
            sample["image"] = self.transform(sample["image"])
            for key in sample.keys():
                if key.startswith("latents_"):
                    sample[key] = self.transform(sample[key])

        return sample
    
    
class WebDataModuleDummyImageNet(pl.LightningDataModule):
    def __init__(self,
                batch_size,
                class_labels,
                val_batch_size=None,
                timestep=0.0,
                num_workers=4,
                val_num_workers=None,
                remove_keys=None,
                transform=None, 
                norm_transform=None, 
        ):
        super().__init__()
        print(f'[WebDataModuleDummyDataset] Setting batch size to {batch_size}')
        
        self.batch_size = batch_size
        self.timestep = timestep
        self.transform = transform
        self.norm_transform = norm_transform
        # -------------------------
        # Dataset loading based on config
        self.class_labels = class_labels
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def make_loader(self, train=True) -> DataLoader:
        # Normalization for experiments in pixel-space
        mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465)
        std = (0.5, 0.5, 0.5) # (0.247, 0.243, 0.261)
        
        if self.norm_transform is None and self.timestep > 1.0:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std)
            ])
        else:
            norm_transform = instantiate_from_config(norm_transform)
        
        # Initialize dummy dataset with on-the-fly generation
        dataset = LatentDummyDataset(
            classes=self.class_labels,
            timestep=self.timestep,
            transform=transform,
            norm_transform=norm_transform
        )
        
        # Create DataLoader with batch size, shuffling, and workers
        return DataLoader(
            dataset,
            batch_size=self.batch_size if train else self.val_batch_size,
            shuffle=train,
            num_workers=self.num_workers if train else self.val_num_workers,
            prefetch_factor=2,  # Overlap data loading with model training
        )

    def train_dataloader(self, timestep: float = 0.0):
        return self.make_loader(train=True)

    def val_dataloader(self, timestep: float = 0.0):
        return self.make_loader(train=False)

    def test_dataloader(self, timestep: float = 0.0):
        return self.make_loader(train=False)







#################################################################################
#                           Simple Testdatasets                                 #
#################################################################################


class DummyDatasetTest(Dataset):
    def __init__(self, num_samples=500000, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.keys_shapes = {k: v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            key: (
                torch.randn(*shape) if len(shape) > 1
                else torch.randint(0, 10, (1,)).squeeze()       # e.g. class labels
            )
            for key, shape in self.keys_shapes.items()
        }


class CIFAR10Test(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__()
        mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465)
        std = (0.5, 0.5, 0.5) # (0.247, 0.243, 0.261)
        
        mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465)
        std = (0.5, 0.5, 0.5) # (0.247, 0.243, 0.261)
        
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std)
            ])
        else:
            transform = instantiate_from_config(transform)

        if target_transform is not None:
            target_transform = instantiate_from_config(target_transform)
        self.dataset = torchvision.datasets.CIFAR10(
            root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return {"image": img, "label": target}
    

class ImageNet256Test(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__()
        mean = (0.5, 0.5, 0.5) # (0.4914, 0.4822, 0.4465)
        std = (0.5, 0.5, 0.5) # (0.247, 0.243, 0.261)
        
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std)
            ])
        else:
            transform = instantiate_from_config(transform)

        if target_transform is not None:
            target_transform = instantiate_from_config(target_transform)
        self.dataset = torchvision.datasets.ImageNet(
            root, split='train' if train else 'val', 
            transform=transform, target_transform=target_transform, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return {"image": img, "label": target}
    






if __name__ == "__main__":
    from omegaconf import OmegaConf
    # config = OmegaConf.load("configs/data/faces.yaml")
    # datamod = WebDataModuleFromConfig(**config["params"])
    # dataloader = datamod.train_dataloader()

    # for i, batch in enumerate(dataloader):
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
    #             print(f"{k:<16}: {v.shape}")
    #         elif isinstance(v, list):
    #             print(f"{k:<16}: {v[:2]}")
    #         else:
    #             print(f"{k:<16}: {type(v)}")
    #     break

    config = OmegaConf.load("configs/data/imagenet256_npy.yaml")

    config["params"]["base_dir"] = "/correct/path/to/your/dataset"
    # Or pass a list of potential paths
    config["params"]["base_dir"] = [
        "/dataset/processed/imagenet-256",
        "./dataset/processed/imagenet-256",
        "../dataset/processed/imagenet-256"
    ]
    
    # Or pass a list of potential paths
    config["params"]["base_dir"] = [
        "/dataset/processed/needs-fix",
        "./dataset/processed/needs-fix",
        "../dataset/processed/needs-fix"
    ]

    datamod = WebDataModuleFromNumpyConfig(**config["params"])
    datamod.setup(stage='fit')
    # config = OmegaConf.load("configs/data/faces.yaml")
    # datamod = WebDataModuleFromConfig(**config["params"])
    # dataloader = datamod.train_dataloader()

    # for i, batch in enumerate(dataloader):
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
    #             print(f"{k:<16}: {v.shape}")
    #         elif isinstance(v, list):
    #             print(f"{k:<16}: {v[:2]}")
    #         else:
    #             print(f"{k:<16}: {type(v)}")
    #     break

    config = OmegaConf.load("configs/data/imagenet256_npy.yaml")

    config["params"]["base_dir"] = "/correct/path/to/your/dataset"
    # Or pass a list of potential paths
    config["params"]["base_dir"] = [
        "/dataset/processed/imagenet-256",
        "./dataset/processed/imagenet-256",
        "../dataset/processed/imagenet-256"
    ]
    
    # Or pass a list of potential paths
    config["params"]["base_dir"] = [
        "/dataset/processed/needs-fix",
        "./dataset/processed/needs-fix",
        "../dataset/processed/needs-fix"
    ]

    datamod = WebDataModuleFromNumpyConfig(**config["params"])
    datamod.setup(stage='fit')
    dataloader = datamod.train_dataloader()

    for i, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print(f"{k:<16}: {v.shape}")
            elif isinstance(v, list):
                print(f"{k:<16}: {v[:2]}")
            else:
                print(f"{k:<16}: {type(v)}")
        break
