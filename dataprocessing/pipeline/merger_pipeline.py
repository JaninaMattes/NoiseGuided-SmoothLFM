import datetime
import os, sys
from matplotlib import pyplot as plt
import numpy as np

from itertools import islice
from omegaconf import OmegaConf
import einops

import torch
import os
from PIL import Image
from pathlib import Path


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Load jutil modules
from jutils.nn import AutoencoderKL
from jutils import instantiate_from_config
from jutils import exists, freeze, default
from jutils import ims_to_grid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)


# Load custom modules
from ldm.flow import FlowModel
from ldm.models.transformer.dit import DiT_models
from data_processing.sampler.data_handler import NumpyDataHandler, HDF5DatasetManager
from data_processing.sampler.data_filter import make_filtered_loader


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
#                       Define Dataloader                   # 
#############################################################
class CustomDataset(Dataset):
    def __init__(self, dataset_dir):
        self.images_dir = Path(dataset_dir) / "Images"
        self.labels_dir = Path(dataset_dir) / "Labels"
        
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")

        self.transform = transforms.ToTensor()

        # Support multiple image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = {}
        for ext in image_extensions:
            for f in self.images_dir.glob(ext):
                image_files[f.stem] = f

        label_files = {f.stem: f for f in self.labels_dir.glob("*.npy")}


        # Keep only matched stems
        self.common_stems = sorted(set(image_files.keys()) & set(label_files.keys()))
        self.image_paths = [image_files[stem] for stem in self.common_stems]
        self.label_paths = [label_files[stem] for stem in self.common_stems]

        print(f"Found {len(self.image_paths)} valid image-label pairs.")


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image) * 2 - 1
            image_tensor = image_tensor.to(torch.float32)

            label = np.load(label_path)
            label_tensor = torch.from_numpy(label).long()

            return {
                "image": image_tensor,
                "label": label_tensor,
                "filename": img_path.stem,
            }
        except Exception as e:
            print(f"[!] Error loading sample {img_path.stem}: {e}")
            return None


class CustomDataLoader(DataLoader):
    def __init__(self, data_folder, batch_size=16, shuffle=True, num_workers=4):
        dataset = CustomDataset(data_folder)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)





#############################################################
#                 Convert the missing Samples               # 
#############################################################
class SamplerPipeline:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        selected_timesteps: list,                       # Timesteps to extract
        first_stage_ckpt = 'checkpoints/sd_ae.ckpt',                               # First stage   (KL Autoencoder)
        second_stage_ckpt = 'checkpoints/SiT-XL-2-256x256.pt',                     # Second stage  (LDM using Flow Matching)#
        input_size: int = 32,                           # Input size
        batch_size: int = 16,
        num_classes: int = 1000,                        # Number of classes
        num_steps: int = 50,                            # Number of steps
        sample_kwargs: dict = None,                     # DDIM sampling kwargs 
        dev: torch.device = None,
        type: str = "train",                            # Type of sampling (sample or encode)
        log_every=1000,                                 # Log every n batches
    ):
        # Device settings
        self.device = dev if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.log_every = log_every

        # Dataset settings
        self.data_dir = dataset_dir
        self.output_dir = output_dir
        self.type = type

        # Sampling settings
        y_null = torch.tensor([self.num_classes] * self.batch_size, device=self.device)
        self.sample_kwargs = sample_kwargs if sample_kwargs else {} 
        self.sample_kwargs.update(  # Add null
            num_steps = num_steps,
            cfg_scale = 1.0,
            uc_cond = y_null,
            cond_key = 'y'
        )     
        self.selected_timesteps = selected_timesteps
        
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

        # Dataloader
        self.dataloader = CustomDataLoader(
            data_folder=self.data_dir,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        
        # Numpy data handler
        self.datahandler = NumpyDataHandler(base_dir=self.output_dir)
        
        
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


    @torch.no_grad()
    def __call__(self):
        """Generate noisy intermediate latents"""        
        # Get selected timesteps
        selected_timesteps = sorted(self.selected_timesteps, reverse=True)
        print(f"Selected timesteps: {selected_timesteps}")
        
        # Get batch
        for batch_idx, batch in enumerate(self.dataloader):
            x = batch['image'].to(self.device).float()
            y = batch['label'].to(self.device).long()
            
        
            # Pipeline
            latent = self.encode_first_stage(x)
            xt, intermediates = self.encode_second_stage(latent.to(self.device), y=y, return_intermediates=True, sample_kwargs=self.sample_kwargs)
            # Generate samples
            intermediates = {f"{t:.1f}": intermediates.get(f"{t:.1f}", None) for t in selected_timesteps}
            intermediates = {k: v for k, v in intermediates.items() if v is not None}
            
            # Plot samples
            if batch_idx % self.log_every == 0:
                print(f"[INFO] Saving samples for batch {batch_idx}...")
                img_file = os.path.join(self.output_dir, f"{self.type}_merger_samples_{batch_idx}.png")
                print(f"Saving samples to {img_file}")
                show_samples(intermediates, split=4, save_to_file=img_file)
            
            data_dict = {
                'image': x.detach().cpu(),
                'latent': xt.detach().cpu(),
                'label': y.detach().cpu(),
                'intermediate_steps': selected_timesteps,
                'intermediates': list(intermediates.values()),
            }

            # Save to Numpy 
            self.datahandler.save_to_numpy(data_dict, group_name=self.type)
            torch.cuda.empty_cache()
        
        print(f"[DONE] Total files saved: {self.datahandler.total_saved}")
        torch.cuda.empty_cache()
        print(f"[INFO] {self.type} pipeline finished.")




if __name__ == "__main__":
    """ Helper pipeline to move the missing samples to the new folder """
    selected_timesteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    type='validation' # 'train'  or 'validation'
    dataset_dir = f"dataset/processed/needs-fix/{type}/"
    output_dir = f"dataset/processed/imagenet-256/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = CustomDataLoader(dataset_dir, batch_size=32)

    # Show the first few batches
    for batch in dataloader:
        if batch is None:
            continue
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        filenames = batch['filename']
        print(filenames[:5])
        break  # just to check loading

    processer = SamplerPipeline(
        selected_timesteps=selected_timesteps,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        first_stage_ckpt='checkpoints/sd_ae.ckpt',
        second_stage_ckpt = 'checkpoints/SiT-XL-2-256x256.pt',                     # Second stage  (LDM using Flow Matching)#
        batch_size=32,
        dev=device,
        type=type, # 'train' or 'validation'
    )

    processer()
    torch.cuda.empty_cache()
    print("Done!")
    