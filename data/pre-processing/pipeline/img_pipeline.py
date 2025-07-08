import datetime
import os, sys
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from itertools import islice
import einops

from tqdm import tqdm
from pathlib import Path

import torch
import os
from PIL import Image
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
class ImgPipeline:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        first_stage_ckpt='checkpoints/sd_ae.ckpt',
        batch_size: int = 16,
        dev: torch.device = None,
    ):
        self.device = dev if dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = dataset_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Create output directory if it doesn't exist
        self.batch_size = batch_size

        # Dataloader
        self.dataloader = CustomDataLoader(
            data_folder=self.data_dir,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

        # Load and freeze the VAE encoder
        first_stage = AutoencoderKL(ckpt_path=first_stage_ckpt).to(self.device)
        self.first_stage = torch.compile(first_stage, fullgraph=True)
        freeze(self.first_stage)
        self.first_stage.eval()

    @torch.no_grad()
    def encode_first_stage(self, x):
        if exists(self.first_stage):
            x = self.first_stage.encode(x)
        return x

    @torch.no_grad()
    def __call__(self):
        
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Batches")):
            try:
                x = batch['image'].to(self.device).float()
                filenames = batch['filename']    
                           
                x1 = self.encode_first_stage(x)
                total_saved = 0
                
                # Iterate over the batch and save each sample
                for sample, filename in zip(x1, filenames):
                    sample = sample.detach().cpu().numpy().astype(np.float32)
                    assert sample.shape == (4, 32, 32), f"Invalid latent shape: {sample.shape}"
                    save_name = os.path.splitext(filename)[0] + ".npy"
                    save_path = os.path.join(self.output_dir, save_name)
                    if os.path.exists(save_path):
                        print(f"File {save_path} already exists. Skipping.")
                        continue
                    # Save the sample
                    np.save(save_path, sample)
                    total_saved += 1
                    print(f"Saved {save_path}")
                
                # Clear the GPU memory
                torch.cuda.empty_cache()
                    
                print(f"[DONE] Total files saved: {total_saved}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")

        torch.cuda.empty_cache()



if __name__ == "__main__":
    """ Helper pipeline to convert images into x1 for completeness """
    
    type='validation' # 'train'  or 'validation'
    dataset_dir = f"dataset/processed/needs-fix/{type}/"
    output_dir = f"dataset/processed/needs-fix/{type}/Latents_1.00"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processer = ImgPipeline(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        first_stage_ckpt='checkpoints/sd_ae.ckpt',
        batch_size=32,
        dev=device
    )

    processer()
    torch.cuda.empty_cache()
    print("Done!")
    