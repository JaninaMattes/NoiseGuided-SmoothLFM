import re
import os, sys
from PIL import Image
import gc

import datetime
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from itertools import islice
import einops

import argparse

import torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.utils import make_grid

# Load jutil modules
from jutils.nn import AutoencoderKL
from jutils import exists, freeze, default
from jutils import ims_to_grid
from jutils import instantiate_from_config

from jutils import denorm
from jutils import ims_to_grid
from jutils.vision import tensor2im
from jutils import tensor2im, ims_to_grid

# Load ldm modules
from ldm.flow import FlowModel
from ldm.models.transformer.dit import DiT_models

from data_processing.tools.norm import denorm_tensor
from data_processing.sampler.data_handler import NumpyDataHandler, HDF5DatasetManager



torch.set_float32_matmul_precision('high')




data_dir = 'dataset/test-samples'

transform = transforms.Compose([
    transforms.Resize(256),                
    transforms.CenterCrop(256),            
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

dataset = ClassIDImageDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for images, labels in dataloader:
    print(images.shape)  # e.g. torch.Size([32, 3, 224, 224])
    print(labels)        # e.g. tensor of class IDs

    # Plotting
    ims = tensor2im(images)        
    ims = ims_to_grid(ims, split=2, channel_last=True)  
    Image.fromarray(ims).show()

    print(f"Min images: {images.min():.2f}, Max images: {images.max():.2f}")
    torch.cuda.empty_cache()
    gc.collect()    



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


def show_samples(intermediates, save_dir=None, prefix="", title="One-sided diffusion: $z_t = (1 - t) z_1 + t \epsilon$"):
    """ Show each sample trajectory as its own grid """
    # Sort timesteps descending (1.0 -> 0.0)
    sorted_intermediates = dict(sorted(intermediates.items(), key=lambda x: float(x[0]), reverse=True))
    
    # [T, B, C, H, W] -> [B, T, C, H, W]
    all_steps = torch.stack(list(sorted_intermediates.values()))  # shape: [T, B, C, H, W]
    all_steps = all_steps.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]

    for i, sample in enumerate(all_steps):  # sample shape: [T, C, H, W]
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
        




""" Data loader """

class ClassIDImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # Extract label from folder name
            match = re.match(r'class_(\d+)', class_folder)
            if not match:
                continue
            label = int(match.group(1))

            # Collect all image paths with label
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, fname)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
    
        
class SampleProcessor:

    def __init__(self, 
                 dataloader,
                 data_dir,
                 hdf5_file_name=None,
                 selected_timesteps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                 first_stage_ckpt='checkpoints/sd_ae.ckpt',
                 second_stage_ckpt='checkpoints/SiT-XL-2-256x256.pt',
                 input_size: int = 32,
                 num_classes: int = 1000,
                 class_labels: list = (250, 236, 291, 292, 294, 296, 339, 340),
                 batch_size: int = 32,
                 num_steps: int = 100,
                 sample_kwargs: dict = None, 
                 device=None,
                 type='test',
                 log_every: int = 10,
                 end_batch_id: int = 999999):
        
        self.device = device 
        self.dataloader = dataloader
        self.hdf5_file_name = hdf5_file_name
        self.data_dir = data_dir
        self.type = type
        self.selected_timesteps = sorted(selected_timesteps, reverse=True)
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.batch_size = batch_size
        self.input_size = input_size
        self.log_every = log_every
        self.end_batch_id = end_batch_id
        
        y_null = torch.tensor([self.num_classes] * self.batch_size, device=self.device)
        self.sample_kwargs = sample_kwargs or {}
        self.sample_kwargs.update({
            'num_steps': num_steps,
            'cfg_scale': 1.0,
            'uc_cond': y_null,
            'cond_key': 'y'
        })
        
        # First stage model
        first_stage = AutoencoderKL(ckpt_path=first_stage_ckpt).to(self.device)
        self.first_stage = torch.compile(first_stage, fullgraph=True)
        freeze(self.first_stage)
        self.first_stage.eval()
        
        # Second stage model
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
        
        self.datahandler = NumpyDataHandler(base_dir=self.data_dir)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage.encode(x) if self.first_stage else x

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage.decode(z) if self.first_stage else z

    @torch.no_grad()
    def encode_second_stage(self, latent, y=None, return_intermediates=True, sample_kwargs=None):
        return self.second_stage.encode(latent, y=y, return_intermediates=return_intermediates, **(sample_kwargs or {}))

    @torch.no_grad()
    def decode_second_stage(self, z, label=None):
        return self.second_stage.generate(z, y=label, **self.sample_kwargs)

    @torch.no_grad()
    def __call__(self):
        print(f"Selected timesteps: {self.selected_timesteps}")
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            x = images.to(self.device).float()
            y = labels.to(self.device).long()
            
            latent = self.encode_first_stage(x)
            xt, intermediates = self.encode_second_stage(latent, y=y, return_intermediates=True, sample_kwargs=self.sample_kwargs)

            filtered_intermediates = {
                f"{t:.1f}": intermediates.get(f"{t:.1f}") for t in self.selected_timesteps if intermediates.get(f"{t:.1f}") is not None
            }

            if batch_idx % self.log_every == 0:
                print(f"Batch {batch_idx}/{self.end_batch_id} - {self.type}")
                show_samples(
                    filtered_intermediates,
                    save_dir=self.data_dir,
                    prefix=f"{self.type}_samples_{batch_idx}"
                )

            
            data_dict = {
                'image': x.detach().cpu(),
                'latent': xt.detach().cpu(),
                'label': y.detach().cpu(),
                'intermediate_steps': self.selected_timesteps,
                'intermediates': list(filtered_intermediates.values()),
            }

            self.datahandler.save_to_numpy(data_dict, group_name=self.type)
            torch.cuda.empty_cache()
            gc.collect()

        # Save to HDF5
        filename = self.hdf5_file_name or f'imagenet256-testset-{datetime.datetime.now().strftime("T%H%M%S")}.hdf5'
        print(f"Save to hdf5 file {filename}.")
        self.save_hdf5(self.data_dir, filename=filename, group_name=self.type)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Processing completed for {self.type} samples. Data saved to {self.data_dir}/{filename}")
        
    def save_hdf5(self, data_dir, filename, group_name='test'):
        hdfhandler = HDF5DatasetManager(data_dir)
        hdfhandler.save_to_hdf5(filename=filename, group_name=group_name)

        hdf5_file = os.path.join(data_dir, filename)
        hdfhandler.print_hdf5_structure(hdf5_file, save_to_file=True)
        print(f"Data saved to {hdf5_file}")

        return hdfhandler.retrieve_from_hdf5(
            file_path=hdf5_file,
            timestep=self.selected_timesteps[0],
            group_name=group_name,
            plot_samples=True
        )



def parse_args():
    parser = argparse.ArgumentParser(description="Process test samples and save latent encodings.")

    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save processed outputs')
    parser.add_argument('--hdf5_file', type=str, default=None, help='Optional HDF5 file name')
    parser.add_argument('--first_stage_ckpt', type=str, required=True, help='Path to first stage checkpoint')
    parser.add_argument('--second_stage_ckpt', type=str, required=True, help='Path to second stage checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--log_every', type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    selected_timesteps = [0.1 * i for i in range(1, 11)]
    class_labels = [0, 1, 88, 89, 96, 154, 236, 250, 236, 291, 292, 294, 296, 339, 340]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = ClassIDImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    processor = SampleProcessor(
        dataloader=dataloader,
        data_dir=args.output_dir,
        hdf5_file_name=args.hdf5_file,
        selected_timesteps=selected_timesteps,
        first_stage_ckpt=args.first_stage_ckpt,
        second_stage_ckpt=args.second_stage_ckpt,
        input_size=args.input_size,
        num_classes=1000,
        class_labels=class_labels,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        sample_kwargs=None,
        device=args.device,
        type=args.type,
        log_every=args.log_every
    )

    processor()
    print("Sample processing completed.")


if __name__ == '__main__':
    main()
