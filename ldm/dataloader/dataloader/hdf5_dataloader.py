import os
import h5py
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import logging 
from typing import Dict, Any, Union
from omegaconf import OmegaConf, ListConfig

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




import h5py
import torch
from torch.utils.data import Dataset
import os
import logging
import threading # Import threading for lock

# Configure logging (optional but good practice)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
class HDF5LatentsDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        group_name: str = 'train',
        source_timestep: float = 0.0,
        target_timestep: float = None,
        lazy_load: bool = True,
        transform=None,
        start_idx: int = 0,
        p: float = 0.5,
    ):
        self.hdf5_file = os.path.abspath(hdf5_file)
        self.group_name = group_name
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        self.transform = transform      # optional: apply only to pixel-space images
        self.lazy_load = lazy_load      # always use lazy_loading (safer for large datasets)
        self.p = p                      # probability of horizontal flip (default: 0.5)
        self.start_idx = start_idx      # default starting index for resuming training

        # --- HDF5 file/group handles initialized to None ---
        self.f = None
        self.group = None
        self.images_dset = None
        self.labels_dset = None
        self.latents_source_dset = None
        self.latents_target_dset = None
        self._lock = threading.Lock()  

        # --- Get dataset size without loading data ---
        try:
            with h5py.File(self.hdf5_file, "r", libver='latest') as temp_f:
                if self.group_name not in temp_f:
                     raise ValueError(f"Group '{self.group_name}' not found in HDF5 file: {self.hdf5_file}")
                temp_group = temp_f[self.group_name]
                if 'images' not in temp_group:
                    raise ValueError(f"Dataset 'images' not found in group '{self.group_name}'")
                self.dataset_size = len(temp_group['images'])
                self.source_latent_key = f"latents_{self.source_timestep:.2f}"
                if self.source_latent_key not in temp_group:
                     raise ValueError(f"Dataset '{self.source_latent_key}' not found in group '{self.group_name}'")

                self.target_latent_key = None
                if self.target_timestep is not None:
                    self.target_latent_key = f"latents_{self.target_timestep:.2f}"
                    if self.target_latent_key not in temp_group:
                        raise ValueError(f"Dataset '{self.target_latent_key}' not found in group '{self.group_name}'")

        except Exception as e:
            logging.error(f"Failed to initialize dataset size from {self.hdf5_file}. Error: {e}")
            raise

        # --- Handle non-lazy loading (load all into memory) ---
        # Discourage lazy loading for large datasets (> 1GB) to avoid memory issues
        self.all_data = None
        if not self.lazy_load:
            logging.warning(f"Loading entire dataset '{self.group_name}' from {self.hdf5_file} into memory.")
            try:
                with h5py.File(self.hdf5_file, "r", libver='latest') as f:
                    group = f[self.group_name]
                    self.all_data = {
                        'images': group['images'][()],
                        'labels': group['labels'][()],
                        self.source_latent_key: group[self.source_latent_key][()]
                    }
                    if self.target_latent_key:
                         self.all_data[self.target_latent_key] = group[self.target_latent_key][()]
            except Exception as e:
                 logging.error(f"Failed to load non-lazy data from {self.hdf5_file}. Error: {e}")
                 raise
            self.dataset_size = len(self.all_data['images'])


    def _open_hdf5(self):
        """Opens the HDF5 file and initializes dataset handles."""
        # This function is called by __getitem__ if self.f is None
        # It's protected by a lock to ensure it runs only once per worker
        try:
            self.f = h5py.File(self.hdf5_file, "r", libver='latest', swmr=False) # swmr=False faster for read-only if not needed
            self.group = self.f[self.group_name]
            self.images_dset = self.group["images"]
            self.labels_dset = self.group["labels"]
            self.latents_source_dset = self.group[self.source_latent_key]
            if self.target_latent_key:
                self.latents_target_dset = self.group[self.target_latent_key]
            # logging.info(f"HDF5 file opened by worker PID: {os.getpid()}")
        except Exception as e:
             logging.error(f"Worker PID {os.getpid()} failed to open HDF5 file {self.hdf5_file}. Error: {e}")
             if self.f: self.f.close() # Release the file handle if it was opened
             self.f = None
             self.group = None
             self.images_dset = None
             self.labels_dset = None
             self.latents_source_dset = None
             self.latents_target_dset = None
             raise # Re-raise the exception


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        # --- Non-lazy path: Use pre-loaded data ---
        if not self.lazy_load and self.all_data:
            image = self.all_data['images'][idx]
            label = self.all_data['labels'][idx]
            latents = self.all_data[self.source_latent_key][idx]
            target_latents = None
            if self.target_latent_key:
                target_latents = self.all_data[self.target_latent_key][idx]

        # --- Lazy path: Read data from HDF5 ---
        else:
            # --- Open HDF5 file if not already open in this worker ---
            if self.f is None:
                with self._lock:            # Acquire the lock to ensure thread safety 
                    if self.f is None:
                        self._open_hdf5()   # Open the file and initialize datasets

            if self.f is None:
                 raise RuntimeError(f"HDF5 file handle is still None in worker {os.getpid()} after attempting to open.")

            # --- Read data for the specific index ---
            try:
                image = self.images_dset[idx]
                label = self.labels_dset[idx]
                latents = self.latents_source_dset[idx]

                target_latents = None
                if self.latents_target_dset:
                    target_latents = self.latents_target_dset[idx]

            except Exception as e:
                 logging.error(f"Error reading index {idx} from {self.hdf5_file} in worker {os.getpid()}. Error: {e}")
                 raise # Re-raise the exception

        # --- Process data (to both lazy and non-lazy) ---
        image = torch.from_numpy(image.copy()).to(torch.float32)
        latents = torch.from_numpy(latents.copy()).to(torch.float32)
        label = torch.tensor(label).to(torch.long) 
        
        if target_latents is not None:
            target_latents = torch.from_numpy(target_latents.copy()).to(torch.float32)

        # Random horizontal flip 
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[-1])        # Flip image
            latents = torch.flip(latents, dims=[-1])
            if target_latents is not None:
                target_latents = torch.flip(target_latents, dims=[-1])

        # Apply norm-transform to image
        if self.transform:
            image = self.transform(image)

       
        sample = {
            'image': image,
            'label': label,
            self.source_latent_key: latents,
        }

        if target_latents is not None:
            sample[self.target_latent_key] = target_latents

        return sample


    """ Retrieve all labels """
    def get_all_labels(self) -> list:
        raw_labels = None
        if not self.lazy_load and self.all_data and self.all_data.get('labels') is not None:
            raw_labels = self.all_data['labels']
        else:
            try:
                with h5py.File(self.hdf5_file, "r", libver='latest') as local_f:
                    if self.group_name in local_f and 'labels' in local_f[self.group_name]:
                        raw_labels = local_f[self.group_name]['labels'][()]
                    else:
                        logging.warning(f"Labels dataset not found for group '{self.group_name}' in {self.hdf5_file} when calling get_all_labels.")
                        return []
            except Exception as e:
                logging.error(f"Error reading labels from HDF5 group '{self.group_name}' in get_all_labels: {e}")
                return []
        
        # Check if raw_labels is None or empty
        if raw_labels is None:
            logging.warning(f"Could not retrieve raw labels for group '{self.group_name}'.")
            return []
        
        # Convert to a flat list of integers
        try:
            if isinstance(raw_labels, np.ndarray):
                return raw_labels.astype(int).flatten().tolist()
            return [int(label) for label in raw_labels]
        except (TypeError, ValueError) as e:
            logging.error(f"Failed to convert labels to a flat list of integers for group '{self.group_name}'. Original data type: {type(raw_labels)}. Error: {e}")
            return []

        
        
    """ Store resume information """
    def state_dict(self):
        return {'start_idx': self.start_idx}

    def load_state_dict(self, state_dict):
        self.start_idx = state_dict['start_idx']
        

    # --- No __del__ needed ---
    # Rely on process termination to close the file handle opened by each worker.
    # Explicitly closing in __del__ is unreliable with multiprocessing.


# ------------------------------------------------------------------------------
""" Pytorch Lightning DataModule from HDF5 Files """
class HDF5DataModule(pl.LightningDataModule):
    def __init__(self,
                    hdf5_file: str,
                    batch_size: int = 32,
                    val_batch_size: int = None,
                    group_name: str = 'train',
                    train: bool = True,
                    validation: bool = False,
                    test: bool = False,
                    source_timestep: float = 0.0,
                    target_timestep: float = None,
                    transform=None,
                    num_workers: int = 8,
                    val_num_workers: int = None,
                    pin_memory: bool = True,
                    prefetch_factor: int = 4,
                    multinode: bool = False,
                    drop_last: bool = False,
                    start_idx: int = 0,
                    balance_classes: bool = True,
                    p: float = 0.5,
                ):
        super().__init__()
        self.hdf5_file = os.path.abspath(hdf5_file)
        self.base_dir = os.path.dirname(self.hdf5_file)
        self.start_idx = start_idx # default starting index for resuming training
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.group_name = group_name
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        self.transform = transform
        self.balance_classes = balance_classes
        self.p = p
        
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.multinode = multinode
        self.drop_last = drop_last
        
        self.train = train
        self.validation = validation
        self.test = test
        
        # Set default values
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def setup(self, stage: str=None):
        """Set up datasets for different stages ('fit', 'validate', 'test', 'predict')."""
        mean = (127.0, 127.0, 127.0)
        std = (127.0, 127.0, 127.0)
        img_transform = transforms.Compose([
            torchvision.transforms.Normalize(mean, std),
        ])

        if (stage == 'fit' or stage is None) and (self.train or self.validation):
            if self.train:
                try:
                    # Check if the file exists
                    if hasattr(self, 'checkpoint') and self.checkpoint is not None:
                        self.start_idx = self.checkpoint['start_idx']  
                        
                    self.train_dataset = HDF5LatentsDataset(
                        hdf5_file=self.hdf5_file,
                        group_name=self.group_name,
                        source_timestep=self.source_timestep,
                        target_timestep=self.target_timestep,
                        transform=img_transform,
                        start_idx=self.start_idx,
                        p=self.p    # Random flip probability for training
                    )
                    logging.info(f"Train dataset setup complete for group '{self.group_name}'")
                    
                    
                    # Class imbalance handling
                    if self.balance_classes:
                        train_labels = self.train_dataset.get_all_labels()
                        if train_labels: 
                            self.train_sampler = self.get_random_weightsampler(train_labels)
                            logging.info(f"WeightedRandomSampler setup for training group.")
                        else:
                            logging.warning(f"No labels found or error in reading labels for group; proceeding without weighted sampling for training.")
                            self.train_sampler = None
            
            
                except Exception as e:
                    logging.error(f"Error setting up train dataset: {e}")
                    raise
            
            if self.validation:
                try:
                    self.val_dataset = HDF5LatentsDataset(
                        hdf5_file=self.hdf5_file,
                        group_name=self.group_name,
                        source_timestep=self.source_timestep,
                        target_timestep=self.target_timestep,
                        transform=img_transform,
                        p=0.0,      # No random flip for validation
                    )
                    logging.info(f"Validation dataset setup complete for group 'validation'")
                except Exception as e:
                     logging.error(f"Error setting up validation dataset: {e}")
                     raise
                
            else:
                logging.warning("Please call setup(fit).")
                return

        if (stage == 'test' or stage is None) and self.test: 
            try:
                self.test_dataset = HDF5LatentsDataset(
                        hdf5_file=self.hdf5_file,
                        group_name=self.group_name,
                        source_timestep=self.source_timestep,
                        target_timestep=self.target_timestep,
                        transform=img_transform,
                        p=0.0,      # No random flip for testing
                )
                logging.info(f"Test dataset setup complete for group 'test'")
            except Exception as e:
                 logging.error(f"Error setting up test dataset: {e}")
                 raise
        else:
            logging.error("Please call setup(fit) or setup(test).")
            return

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Did you call `setup(stage='fit')` and enable training?")

        shuffle_flag = False if self.balance_classes else True
        train_data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            drop_last=self.drop_last,
            sampler=self.train_sampler if self.balance_classes else None
        )
        return train_data

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Did you call `setup(stage='fit')` and enable validation?")
        
        val_data = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.val_num_workers > 0 else None,
            persistent_workers=self.val_num_workers > 0,
            drop_last=False
        )
        return val_data

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Did you call `setup(stage='test')` and enable testing?")

        test_data = DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.val_num_workers > 0 else None,
            persistent_workers=self.val_num_workers > 0,
            drop_last=False
        )
        return test_data


    def get_class_weights(labels: list):
        '''Use binning approach. 
            https://discuss.pytorch.org/t/sampling-with-replacement/26474/19
        '''
        target = torch.LongTensor(labels)
        # class_sample_count = torch.tensor([(target < 2 or target > 2).sum(), (target == 2).sum()])
        class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
        weight = 1. / class_sample_count.float()
        return torch.tensor([weight[t] for t in target])



    def get_random_weightsampler(self, target_labels: list):
        """ Return a WeightedRandomSampler to correct for class imbalance. """
        if not target_labels:
            logging.warning("Target labels list is empty. Cannot create sampler.")
            return None

        target_np = np.array(target_labels)
        unique_labels, counts = np.unique(target_np, return_counts=True)
        logging.info(f"Unique labels: {unique_labels}, Counts: {counts}")

        # Handle edge case: zero counts
        weights_per_class = np.array([
            len(target_np) / (len(counts) * c) if c > 0 else 0.0 for c in counts
        ])
        logging.info(f"Weights per class: {weights_per_class}")
        
        # Map each label to its weight
        label_to_weight = dict(zip(unique_labels, weights_per_class))
        sample_weights = np.array([label_to_weight[label.item()] for label in target_np], dtype=np.float64)

        # Check weights
        if np.all(sample_weights <= 0):
            logging.warning("All sample weights are zero. Returning None.")
            return None

        # Approach 1:
        weights_tensor = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

        # Approach 2:
        # samples_weight = get_class_weights(dataset.get_all_labels())
        # sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(dataset), replacement=True)
        return sampler

    
    

if __name__ == "__main__":
    # Sample test 
    print("--"*20)
    print("Testing with standard test-dataset")
    test_hdf5_file = "./dataset/processed/testset-256/imagenet256-test-dataset-T110525.hdf5"
    datamodule = HDF5DataModule(
        hdf5_file=test_hdf5_file,
        group_name='test',
        batch_size=32,
        source_timestep=0.0,
        train=False,
        test=True
    )
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    for batch in test_loader:
        print(batch)
        break
    
    
    print("--"*20)
    print("Testing with unbalanced classes")
    # Example usage
    hdf5_file = "./dataset/processed/needs-fix/imagenet256_data-T120631.hdf5" 
    datamodule = HDF5DataModule(
        hdf5_file=hdf5_file,
        batch_size=32,
        group_name='train',
        source_timestep=0.0,
        train=True,
        validation=False,
        balance_classes=False,
        p=0.5,
    )
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        labels = batch['label']
        print(labels)
        break
    
    
    print("--"*20)
    print("Testing with balancde classes")
    # Example usage
    hdf5_file = "./dataset/processed/needs-fix/imagenet256_data-T120631.hdf5" 
    hdf5_file = './dataset/processed/needs-fix/imagenet256-dataset-T000001.hdf5'
    datamodule = HDF5DataModule(
        hdf5_file=hdf5_file,
        batch_size=32,
        group_name='train',
        source_timestep=0.0,
        train=True,
        validation=False,
        balance_classes=True,
        p=0.5,
    )
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        labels = batch['label']
        print(labels)
        break