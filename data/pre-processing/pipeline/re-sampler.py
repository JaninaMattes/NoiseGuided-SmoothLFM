import datetime
import os, sys

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from itertools import islice
import einops

import random 
from pathlib import Path

import torch
import torchvision.transforms as T
import torch.nn.functional as F


# Load jutil modules
from jutils.nn import AutoencoderKL
from jutils import instantiate_from_config
from jutils import exists, freeze, default
from jutils import ims_to_grid

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)

# Load custom modules
from ldm.flow import FlowModel
from ldm.models.transformer.dit import DiT_models
from data_processing.sampler.data_handler import NumpyDataHandler, HDF5DatasetManager
from data_processing.sampler.data_filter import make_filtered_loader
from data_processing.tools.norm import denorm_tensor

# Set precision to high
torch.set_float32_matmul_precision('high')




#############################################################
#                          Utils                           # 
#############################################################

def img_to_grid(img, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and img.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        img = einops.rearrange(img, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
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


def show_samples(intermediates, split=4, save_to_file=None):
    """ Show samples """
    intermediates = dict(sorted(intermediates.items(), key=lambda x: float(x[0]), reverse=True))  # Sort by timestep
    ims = torch.stack(list(intermediates.values()), dim=1)
    ims = einops.rearrange(ims, "t b c h w -> (t b) c h w")
    ims = un_normalize_img(ims)
    ims_grid = ims_to_grid(ims, stack="row", split=split)

    plt.imshow(ims_grid.cpu().numpy())
    plt.title(r"Forward Diffusion $x_1 \rightarrow x_0$")
    plt.axis("off")
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight')
    plt.show()
    plt.close() 





#############################################################
#                 Pipelione Latent Sampler                   # 
#############################################################

class SampleProcessor:
    def __init__(
        self, 
        selected_timesteps: list,                       # Timesteps to extract
        dataset_cfg: str,                              # Dataset config
        dataset_dir: str,                               # Dataset directory
        hdf5_dir: str,                                  # HDF5 directory
        first_stage_ckpt = 'checkpoints/sd_ae.ckpt',    # First stage   (KL Autoencoder)
        second_stage_ckpt = 'checkpoints/SiT-XL-2-256x256.pt',                     # Second stage  (LDM using Flow Matching)#
        start_batch_id: int = 0,                        # Starting batch ID
        end_batch_id: int = 10000,                      # Ending batch ID
        input_size: int = 32,                           # Input size
        num_classes: int = 1000,                        # Number of classes
        class_labels: list = None,                      # Class labels to filter
        batch_size: int = 32,                           # Batch size
        num_steps: int = 100,                            # Number of steps
        sample_kwargs: dict = None,                     # DDIM sampling kwargs 
        dev: torch.device = None,                       # Device to use for sampling
        type: str = "train",                            # Type of sampling (sample or encode)
        log_every: int = 1000,                                 # Log every n batches
        filtered_loader: bool = True,                   # Use filtered loader
        ):
        
        # Device settings
        self.device = dev if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.start_batch_id = start_batch_id
        self.end_batch_id = end_batch_id
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.log_every = log_every

        # Dataset settings
        self.data_dir = dataset_dir
        self.hdf5_dir = hdf5_dir
        self.dataset_cfg = dataset_cfg
        self.type = type

        # Sampling settings
        y_null = torch.tensor([self.num_classes] * self.batch_size, device=self.device)
        self.sample_kwargs = sample_kwargs if sample_kwargs else {} 
        self.sample_kwargs.update(  # Add null
            num_steps = num_steps,
            cfg_scale = 1.0,    # unconditional sampling
            uc_cond = y_null,
            cond_key = 'y'
        )     
        self.selected_timesteps = selected_timesteps
        if not self.selected_timesteps:
          raise ValueError("No timesteps provided. Please specify --timesteps.")
 
        # KL Autoencoder - first stage settings
        first_stage = AutoencoderKL(ckpt_path=first_stage_ckpt).to(self.device)
        self.first_stage = torch.compile(first_stage, fullgraph=True)
        freeze(self.first_stage)
        self.first_stage.eval()
            
        # Second stage (Flow with SiT) settings
        flow_model = net = DiT_models["DiT-XL/2"](
                input_size=self.input_size,
                num_classes=self.num_classes,
                learn_sigma=True,               # we "learn sigma" but never use it in SiT/DiT
                load_from_ckpt=second_stage_ckpt,
            ).to(self.device)
        flow_model = FlowModel(net, schedule="linear").to(self.device)
        self.second_stage = torch.compile(flow_model, fullgraph=True)
        freeze(self.second_stage)
        self.second_stage.eval()      


        # Load data
        cfg_path = dataset_cfg
        dataset_cfg = OmegaConf.load(cfg_path)
        self.datamod = instantiate_from_config(dataset_cfg)
        assert class_labels is not None, "Please provide class labels to filter the dataset."
        
        if filtered_loader:
            if self.type == "train":
                self.datamod.train_dataloader = lambda: make_filtered_loader(
                    data=self.datamod, data_cfg=self.datamod.train, class_labels=class_labels, train=True, batch_size=self.batch_size)
                print(f"Using filtered loader for training with class labels: {class_labels}")
            elif self.type == "validation":
                self.datamod.val_dataloader = lambda: make_filtered_loader(
                    data=self.datamod, data_cfg=self.datamod.validation, class_labels=class_labels, train=False, batch_size=self.batch_size)
                print(f"Using filtered loader for validation with class labels: {class_labels}")
            else:
                raise ValueError(f"Unknown type {self.type}. Please use 'train' or 'validation'.")
        
        self.datamod.setup('fit')
                
        """ Data handler """
        self.datahandler = NumpyDataHandler(base_dir=self.data_dir)
    
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if exists(self.first_stage):
            x = self.first_stage.encode(x)
        return x
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        if exists(self.first_stage):
            z = self.first_stage.decode(z)
        return z
    
    @torch.no_grad()
    def encode_second_stage(self, latent, y=None, return_intermediates=True, sample_kwargs=None):
        """ Forward diffusion """
        if exists(self.second_stage):
            xt, intermediates = self.second_stage.encode(latent, y=y, return_intermediates=return_intermediates, **(sample_kwargs or {}))               # x0: noise, x: target, t: timestep
        return xt, intermediates
    
    @torch.no_grad()
    def decode_second_stage(self, z, label=None):
        """ Euler sampling """
        if exists(self.second_stage):
            z = self.second_stage.generate(z, y=label, **self.sample_kwargs)
        return z
  
    
    # Fix class method
    @staticmethod
    def check_latent_exists(latent_dir, image_name, timestep, filetype='.npy'):
        latent_path = os.path.join(latent_dir, f"Latents_{timestep}", f"{image_name}{filetype}")
        print(f"Checking if latent exists at: {latent_path}")
        return os.path.exists(latent_path)


    @torch.no_grad()
    def __call__(self):
        """Generate noisy latents"""

        # Setup dataloader
        dataloader = self.datamod.train_dataloader() if self.type == 'train' else self.datamod.val_dataloader()

        base_dir = Path(self.data_dir) / self.type
        images_dir = base_dir / "Images"
        labels_dir = base_dir / "Labels"

        self.selected_timesteps = sorted(self.selected_timesteps, reverse=True)
        print(f"Selected timesteps: {self.selected_timesteps}")

        image_files = list(images_dir.glob("*.png"))

        for image_file in image_files:
            image_name = image_file.stem
            label_file = labels_dir / f"{image_name}.npy"
            if not label_file.exists():
                raise FileNotFoundError(f"Label file {label_file} does not exist.")

            missing_latent = False
            for t in self.selected_timesteps:
                latent_file = base_dir / f"Latents_{t:.2f}" / f"{image_name}.npy"
                if not latent_file.exists():
                    print(f"Missing: {latent_file}")
                    missing_latent = True
                    break

            if missing_latent:
                print(f"Regenerating data for index {image_name}")
                self.generate_samples(base_dir, file_index=image_name, sample_size=1, dataloader=dataloader)



    @torch.no_grad()
    def generate_samples(self, base_dir, file_index, sample_size=1, dataloader=None):
        """Generate noisy latents"""
        if self.type == 'train':
            dataloader = self.datamod.train_dataloader()
        else:
            dataloader = self.datamod.val_dataloader()

        if dataloader is None:
            raise ValueError("Dataloader is None. Please provide a valid dataloader.")

        sample = next(iter(dataloader))  # or use random batch sampling logic
        if sample_size == 1:
            idx = random.randint(0, len(sample['image']) - 1)
            x = sample['image'][idx].unsqueeze(0).to(self.device)
            y = sample['label'][idx].unsqueeze(0).to(self.device)
        else:
            indices = random.sample(range(len(sample['image'])), sample_size)
            x = sample['image'][indices].to(self.device)
            y = sample['label'][indices].to(self.device)

        print(f"Processing sample shape: {x.shape}, label shape: {y.shape}")

        latent = self.encode_first_stage(x)
        xt, intermediates = self.encode_second_stage(latent, y=y, return_intermediates=True, sample_kwargs=self.sample_kwargs)

        intermediates = {f"{t:.1f}": intermediates.get(f"{t:.1f}", None) for t in self.selected_timesteps}
        intermediates = {k: v for k, v in intermediates.items() if v is not None}

        data_dict = {
            'image': x.detach().cpu(),
            'latent': xt.detach().cpu(),
            'label': y.detach().cpu(),
            'intermediate_steps': self.selected_timesteps,
            'intermediates': list(intermediates.values()),
        }

        self.save_to_numpy(base_dir, file_index, data_dict, group_name=self.type)
        torch.cuda.empty_cache()


        
    
    def save_to_numpy(self, base_dir, file_index, data: dict, group_name: str = 'train'):
        group_dir = base_dir
        img_dir = group_dir / 'Images'
        labels_dir = group_dir / 'Labels'

        file_index = str(file_index)
        print("=" * 50)
        print(f"Saving data for file index: {file_index}")

        if 'image' in data:
            self.save_images(data['image'], img_dir, file_index)
        if 'label' in data:
            self.save_labels(data['label'], labels_dir, file_index)
        if 'latent' in data:
            latent_dir = group_dir / 'Latents_0.00'
            self.save_latents(data['latent'], latent_dir, file_index)

        if 'intermediate_steps' in data and 'intermediates' in data:
            for step, interm_batch in zip(data['intermediate_steps'], data['intermediates']):
                if step == 0.0:
                    continue
                latent_dir = group_dir / f'Latents_{step:.2f}'
                self.save_latents(interm_batch, latent_dir, file_index)

                

    def save_images(self, images: np.ndarray, img_dir: Path, file_index: int) -> None:
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
            image_path = img_dir / f'{file_index}.png'
            os.makedirs(img_dir, exist_ok=True)
            try:
                # store unnormalized
                img = transform(img)
                img.save(str(image_path))
                print(f"Shape of image: {img.size}")
                print(f"Saved image to: {image_path}")
            except Exception as e:
                logging.error(f"Error saving image {image_path}: {e}")
                raise
            
    
    def save_latents(self, latents: np.ndarray, latent_dir: Path, file_index: int) -> None:
        """Save batch of latent representations as numpy files."""
        # Convert to numpy array
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()

        for i, latent in enumerate(latents):
            latent_path = latent_dir / f'{file_index}.npy'
            os.makedirs(latent_dir, exist_ok=True)  # Ensure directory exists
            try:
                # Store unnormalized with single precision
                latent = latent.astype(np.float32)
                np.save(str(latent_path), latent)
                print(f"Shape of latent: {latent.shape}")
                print(f"Saved latent to: {latent_path}")
            except Exception as e:
                logging.error(f"Error saving latent {latent_path}: {e}")
                raise
    
    
    def save_labels(self, labels: np.ndarray, labels_dir: Path, file_index: int) -> None:
        """Save labels as individual numpy files with double precision."""
        labels = labels.detach().cpu().numpy()
        for i, label in enumerate(labels):
            label_path = labels_dir / f'{file_index}.npy'
            os.makedirs(labels_dir, exist_ok=True)
            try:
                # Double precision
                label = label.astype(np.float64)
                np.save(str(label_path), label)
                print(f"Label: {label}")
                print(f"Shape of label: {label.shape}")
                print(f"Saved label to: {label_path}")
            except Exception as e:
                logging.error(f"Error saving label {label_path}: {e}")
                raise
            
            




import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Sample Generator for Diffusion Model Datasets")
    parser.add_argument('--dataset_dir', type=str, default='dataset/processed/imagenet-256')
    parser.add_argument('--dataset_cfg', type=str, default='configs/data/imagenet256_mvl.yaml')
    parser.add_argument('--timesteps', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--class_labels', type=int, nargs='+', default=[
            0, 1, 84, 87, 88, 89, 90, 92, 93, 94, 95, 96, 99, 100, 105, 106, 130, 144, 145, 152, 
            153, 154, 158, 172, 176, 207, 208, 219, 231, 232, 234, 236, 237, 248, 249, 250, 251, 
            254, 258, 259, 260, 263, 264, 269, 270, 271, 277, 278, 279, 280, 282, 283, 284, 288, 
            289, 290, 291, 292, 293, 294, 295, 296, 321, 322, 323, 324, 330, 331, 332, 339, 340, 
            344, 346, 347, 348, 349, 350, 352, 353, 354, 361, 362, 365, 366, 368, 383, 387, 388, 
            954, 957
    ]) # V2 dataset class labels
    parser.add_argument('--batch_size', type=int, default=256)  # Batch size for sampling
    parser.add_argument('--start_batch_id', type=int, default=0)
    parser.add_argument('--end_batch_id', type=int, default=10)
    parser.add_argument('--split', type=str, choices=['train', 'validation'], default='train')
    parser.add_argument('--hdf5_file', type=str, default=None)
    parser.add_argument('--filtered_loader', type=bool, default=True, 
                        help="Use filtered loader to only sample specific class labels from the dataset.")

    args = parser.parse_args()

    print(f"Selected timesteps: {args.timesteps}")   
    class_labels = sorted(args.class_labels)
    print(f"Class labels: {class_labels}")
    print(f"Number of class labels: {len(class_labels)}")

    processer = SampleProcessor(
        selected_timesteps=args.timesteps,
        dataset_cfg=args.dataset_cfg,
        dataset_dir=args.dataset_dir,
        hdf5_dir=args.hdf5_file,
        start_batch_id=args.start_batch_id,
        end_batch_id=args.end_batch_id,
        num_classes=1000,
        class_labels=class_labels,
        batch_size=args.batch_size if args.batch_size > 0 else len(class_labels),
        log_every=5000,
        type=args.split,
        filtered_loader=args.filtered_loader,
    )
    processer()
    print("Sample processing completed.")
    
    # processer.save_hdf5(args.dataset_dir, file_index='imagenet256-dataset-T000003.hdf5', group_name=args.split)
    # CUDA_VISIBLE_DEVICES=1 python '/export/home/ra93jiz/dev/Img-IDM/data_processing/sampler/sample_processor.py' --start_batch_id 0 --end_batch_id 1800 --split train --dataset_dir 'dataset/processed/imagenet-256' --batch_size 32