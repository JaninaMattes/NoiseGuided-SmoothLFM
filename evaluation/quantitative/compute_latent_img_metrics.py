# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import os, sys
import csv
import gc
from tqdm import tqdm
import shutil
from pathlib import Path


import time
import torch
import gc
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from lightning import seed_everything
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from scipy import linalg

from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import rcParams


# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from elatentlpips import ELatentLPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from pytorch_fid.inception import InceptionV3

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

import torch_fidelity
from prdc import compute_prdc
from pytorch_fid.inception import InceptionV3

# Jutils 
from jutils import denorm
from jutils import ims_to_grid
from jutils.vision import tensor2im
from jutils import exists, freeze, default
from jutils import tensor2im, ims_to_grid

# Setup project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
sys.path.append(project_root)


# Custom modules
from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule
from ldm.utils.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor # denorm tensor -- just for plotting


torch.set_float32_matmul_precision('high')




def save_tensor_as_png(tensor: torch.Tensor, path: str):
    """
    Save a torch tensor (C, H, W) as a .png image.
    Assumes tensor is in [-1, 1] or [0, 1].
    """
    tensor = denorm_tensor(tensor).detach().cpu() # just for visualization
    img = TF.to_pil_image(tensor.byte())
    img.save(path)


def load_png_to_tensor(path: str) -> torch.Tensor:
    """
    Load a .png image and convert to torch.Tensor in [0, 1], shape [3, H, W].
    """
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # Returns float tensor in [0, 255]


def load_png_to_tensor_normalized(path: str) -> torch.Tensor:
    """
    Load .png image and return tensor in [-1, 1], shape [3, H, W].
    """
    tensor = load_png_to_tensor(path)  # [0, 255]
    tensor = tensor / 127.5 - 1.0  # [0, 255] → [-1, 1]
    return tensor




############################################
#   Utils
############################################
from filelock import FileLock

def clear_folder(folder_path: Path):
    lock_file = folder_path.with_suffix('.lock')
    with FileLock(lock_file):
        try:
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"[INFO] Removed existing folder: {folder_path}")
            elif folder_path.exists() and folder_path.is_file():
                folder_path.unlink()
                print(f"[INFO] Removed existing file: {folder_path}")
            else:
                print(f"[INFO] Folder {folder_path} does not exist, nothing to clear.")
        except Exception as e:
            print(f"[ERROR] Failed to clear folder {folder_path}: {e}")
            raise RuntimeError(f"[ERROR] Failed to clear folder {folder_path}: {e}")
        finally:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created folder: {folder_path}")
            

    


############################################
# Image metrics tracker
############################################
class LatentImageMetricsTracker(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()
        self.e_lpips = ELatentLPIPS(encoder="sd15", augment="bg").to(self.device).eval()

        self.reset()

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.maes = []
        self.cossims = []
        self.e_lpips_scores = []
        
        
    @torch.no_grad()
    def update(self, target, pred):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Normalize pred and target for pixel metrics [0, 1]
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float').to(self.device)
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float').to(self.device)

        # Cosine similarity computation
        B = pred_norm.shape[0]
        pred_flat = F.normalize(pred_norm.view(B, -1), dim=1)  # shape: (B, C*H*W)
        target_flat = F.normalize(target_norm.view(B, -1), dim=1)
        cossim = torch.sum(pred_flat * target_flat, dim=1)  # shape: (B,)
        self.cossims.append(cossim.detach().cpu())
        
        # Standard metrics
        self.ssims.append(self.ssim(pred_norm, target_norm).detach().cpu())
        self.psnrs.append(self.psnr(pred_norm, target_norm).detach().cpu())
        self.mses.append(torch.mean((pred_norm - target_norm) ** 2, dim=[1, 2, 3]).detach().cpu())
        self.maes.append(torch.mean(torch.abs(pred_norm - target_norm), dim=[1, 2, 3]).detach().cpu())
        self.e_lpips_scores.append(self.e_lpips(pred_norm * 2 - 1, target_norm * 2 - 1).detach().cpu())

        # Clean up memory
        del pred_norm, target_norm
        torch.cuda.empty_cache()

    @torch.no_grad()
    def aggregate(self):
        return dict(
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.e_lpips_scores).mean().item(),
            cossim=torch.cat(self.cossims).mean().item()
        )







#########################################################
#                 Image Synthesis for Samples           #
#########################################################
import torch
import os
import gc
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from matplotlib import rcParams


def generate_samples(
    beta_vae_module,
    gt_imgs,
    xt_latent,
    labels=None,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    num_steps=50,
    num_classes=1000,
    denorm_fn=None,  # If needed later
    plot_samples=False,
    nrow=8,
    title="Generated Samples",
    save_path=None,
    resize_to=128,
    device=None,
):
    device = device or beta_vae_module.device

    # Move tensors to device
    gt_imgs = gt_imgs.to(device)
    xt_noise_latents = xt_latent.to(device)
    
    if labels is not None:
        labels = labels.to(device).squeeze()

    with torch.no_grad():
        # Encode-Decode with ß-VAE
        latents = beta_vae_module.model.encode(xt_noise_latents)['latent_dist'].sample()
        fake_noise_latents = beta_vae_module.model.decode(latents)['sample']

    # Plotting (optional)
    if plot_samples:
        
        # Decode to RGB space
        latent = beta_vae_module.decode_second_stage(fake_noise_latents, label=labels)
        fake_gt_imgs = beta_vae_module.decode_first_stage(latent)
        real_gt_imgs = gt_imgs  # Already unnormalized

        real_gt_imgs_ = denorm_tensor(real_gt_imgs).detach().cpu()
        fake_gt_imgs_ = denorm_tensor(fake_gt_imgs).detach().cpu()
        real_gt_imgs_resized = TF.resize(real_gt_imgs_, [resize_to, resize_to])
        fake_gt_imgs_resized = TF.resize(fake_gt_imgs_, [resize_to, resize_to])

        interleaved = []
        for real, fake in zip(real_gt_imgs_resized, fake_gt_imgs_resized):
            interleaved.extend([real, fake])

        grid = make_grid(torch.stack(interleaved), nrow=nrow, padding=0)

        rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
        fig, ax = plt.subplots(figsize=(grid.shape[2] / 50, grid.shape[1] / 50))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
        ax.set_title(title, fontsize=14)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"[INFO] Saved sample plot to {save_path}")

        plt.show()
        plt.close(fig)

        del grid, fake_gt_imgs_, real_gt_imgs_, real_gt_imgs_resized, fake_gt_imgs_resized
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    return fake_noise_latents.detach().cpu(), xt_noise_latents.detach().cpu()




    

##################################################
# Get dataloader by group
##################################################
def get_dataloader_by_group(data_module, group: str):
    if group == "validation":
        print("[INFO] Using validation dataloader")
        return data_module.val_dataloader()
    elif group == "test":
        print("[INFO] Using test dataloader")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Unsupported group: {group}")




def get_last_valid_sample_index(fake_dir: Path, real_dir: Path, label_dir: Path) -> int:
    fake_ids = {int(f.stem) for f in fake_dir.glob("*.png") if f.stem.isdigit()}
    real_ids = {int(f.stem) for f in real_dir.glob("*.png") if f.stem.isdigit()}
    label_ids = {int(f.stem) for f in label_dir.glob("*.npy") if f.stem.isdigit()}

    common_ids = fake_ids & real_ids & label_ids
    if not common_ids:
        return 0                # No labels found

    return max(common_ids) + 1  # Resume from next index






@torch.no_grad()
def run_data_collection_and_evaluation(
    checkpoint,
    data_path,
    project_name,
    model_name,
    group="validation",
    source_timestep=0.50,
    target_timestep=1.00,
    beta=0.1,
    samples_per_class=10,
    num_pairs=5,
    num_interpolations=16,
    cfg_scales=[1.0],
    ccfg_scales=[1.0],
    batch_size=32,
    dataset_name="imagenet256-testset",
    root_path="results",
    max_samples=50000,
    num_steps=50,
    num_classes=1000,
    plot_every_n_batches=1000,
    device=None,
):      
    # Set device
    seed_everything(2025)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(checkpoint, map_location="cpu")
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

    print(f"[INFO] Model loaded with {sum(p.numel() for p in beta_vae_module.parameters()) / 1e6:.2f}M parameters.")
    print(f"[INFO] Group: {group}, Dataset: {dataset_name}")

    # Load data
    data = HDF5DataModule(
        hdf5_file=data_path,
        batch_size=batch_size,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        num_workers=4,
        train=False,
        validation=(group == "validation"),
        test=(group == "test"),
        group_name=group,
    )
    data.setup(stage="fit" if group == "validation" else "test")
    dataloader = get_dataloader_by_group(data, group)

    # Setup results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_dir = Path(root_path) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results directory: {base_results_dir}")

    # CSV setup
    csv_path = base_results_dir / f"{model_name}_metrics.csv"
    csv_header = [
        "cfg_scale",
        "ccfg_scale",
        "ssim",
        "lpips",
        "psnr",
        "mse",
        "mae",
        "cossim",
        "elapsed_time",
        "count",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    metrics_records = []
    count = 0

    for cfg_scale, ccfg_scale in zip(cfg_scales, ccfg_scales):
        img_metrics_tracker = LatentImageMetricsTracker(device=device)

        for batch in tqdm(dataloader, desc=f"[CFG={cfg_scale}, CCFG={ccfg_scale}]"):
            if count >= max_samples:
                print(f"[INFO] Reached max_samples = {max_samples}. Stopping early.")
                break

            gt_imgs = batch["image"].to(device)
            gt_latent = batch[f"latents_{source_timestep:.2f}"].to(device)
            labels = batch["label"].to(device)

            # Time measurement
            start_time = time.time()

            fake_latent_imgs, _ = generate_samples(
                beta_vae_module=beta_vae_module,
                gt_imgs=gt_imgs,
                xt_latent=gt_latent,
                labels=labels,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                num_steps=num_steps,
                num_classes=num_classes,
                plot_samples=(count % plot_every_n_batches == 0),
                save_path=base_results_dir,
                device=device,
            )

            elapsed_time = time.time() - start_time
            print(f"[BATCH] Time taken for generation: {elapsed_time:.2f} seconds")

            img_metrics_tracker.update(gt_latent, fake_latent_imgs)
            batch_metrics = img_metrics_tracker.aggregate()
            img_metrics_tracker.reset()

            count += gt_latent.size(0)  # Increment count by batch size

            record = {
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "ssim": batch_metrics["ssim"],
                "lpips": batch_metrics["lpips"],
                "psnr": batch_metrics["psnr"],
                "mse": batch_metrics["mse"],
                "mae": batch_metrics["mae"],
                "cossim": batch_metrics["cossim"],
                "elapsed_time": elapsed_time,
                "count": count,
            }

            metrics_records.append(record)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(list(record.values()))

            print(
                f"[BATCH] CFG={cfg_scale}, CCFG={ccfg_scale} | "
                f"SSIM={record['ssim']:.4f}, LPIPS={record['lpips']:.4f}, "
                f"PSNR={record['psnr']:.4f}, MSE={record['mse']:.4f}, "
                f"MAE={record['mae']:.4f}, COSSIM={record['cossim']:.4f}"
            )

        # Cleanup
        del img_metrics_tracker
        torch.cuda.empty_cache()
        gc.collect()
        
        
    # Final aggregated metrics
    df_metrics = pd.DataFrame(metrics_records)

    summary_row = {
        "cfg_scale": "Average",
        "ccfg_scale": "Average",
        "ssim": df_metrics["ssim"].mean(),
        "lpips": df_metrics["lpips"].mean(),
        "psnr": df_metrics["psnr"].mean(),
        "mse": df_metrics["mse"].mean(),
        "mae": df_metrics["mae"].mean(),
        "cossim": df_metrics["cossim"].mean(),
        "elapsed_time": df_metrics["elapsed_time"].sum(),
        "count": count,
    }

    agg_csv_path = base_results_dir / f"{model_name}_metrics_summary_{timestamp}.csv"
    pd.DataFrame([summary_row]).to_csv(agg_csv_path, index=False)

    print(f"[INFO] Aggregated results saved to {agg_csv_path}")
    print(summary_row)

    # Cleanup
    del beta_vae_module, dataloader, data
    torch.cuda.empty_cache()
    gc.collect()

    return summary_row, base_results_dir





if __name__ == "__main__":
    
    #####################################
    # Evaluation Parameters
    #####################################
    # Model checkpoints
    source_timestep     = 0.50
    target_timestep     = 1.00
    beta                = 0.1     # Beta value for the VAE
    dataset_name        = 'imagenet256-testset-T151412'
    group               = "test"  # "validation" or "test"
    root_path           = './results'  # Root directory for results
    baseline            = (source_timestep == 0.50 and target_timestep == 0.50)
    batch_size          = 24
    samples_per_class   = 12
    num_pairs           = 10
    num_interpolations  = 18
    cfg_scale           = 4.0
    ccfg_scale          = 1.0
    max_samples         = 100   # 50000
    num_steps           = 50
    num_classes         = 1000
    plot_every_n_batches = 100  # Plot every N batches
    
    #####################################
    # Model Paths for SiT-XL-2
    #####################################
    
    # beta: 1e-4 
    Beta02x10x_1e4b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'
    
    # beta: 0.1
    Beta00x00x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt'
    Beta02x02x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt'                     # Open 
    Beta05x05x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt'                     # Open (Baseline)
    Beta05x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-02/101646/checkpoints/last.ckpt'                                       # Open
    Beta04x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.40x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'  
    Beta03x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.30x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'  
    Beta02x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt'                    ####### DONE
    Beta00x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-1.00x-0.1b/2025-06-18/29852/V0-eV2/2025-06-24/29852/checkpoints/last.ckpt'                 # Open

    # beta: 0.5
    Beta02x10x_05b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.5b/2025-06-30/manual/V0/2025-07-19/101646/checkpoints/last.ckpt'
    Beta05x10x_05b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-31/101646/checkpoints/last.ckpt'                                                                 # Open (Baseline)

    # beta: 1.0
    Beta05x05x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
    Beta05x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'                                                                                                                                   # Open
    Beta02x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'                                          # Open

    # beta: 2.0
    Beta02x10x_2b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-2.0b/V2/2025-07-16/101646/checkpoints/last.ckpt'                     # Open
    Beta05x10x_2b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-2b/2025-06-30-1435/manual/V2/2025-07-31/101646/checkpoints/last.ckpt'  # Open (Baseline)


    # beta: 3.0
    Beta02x10x_3b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt'                     # Open

    # beta: 5.0
    Beta05x05x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
    Beta05x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'                                                                                                                                  # Open
    Beta02x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'                   # Open

    
    #####################################
    # Dataset & Evaluation Parameters
    #####################################
    test_data_path               = './dataset/processed/testset-256/imagenet256-testset-T222343.hdf5' #'./dataset/processed/testset-256/imagenet256-testset-T151412.hdf5' # ./dataset/processed/testset-256/imagenet256-testset-T151633.hdf5'
    validation_data_path         = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'

    model_configs = [
        # beta: 1e-4
        {
            "checkpoint": Beta02x10x_1e4b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.0001,
            "group_name": "Beta1e-4"
        },

        # beta: 0.1
        {
            "checkpoint": Beta00x00x_01b,
            "source_timestep": 0.00,
            "target_timestep": 0.00,
            "beta": 0.1,
            "group_name": "Beta0.1_Reconstruction"
        },
        {
            "checkpoint": Beta02x02x_01b,
            "source_timestep": 0.20,
            "target_timestep": 0.20,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": Beta05x05x_01b,
            "source_timestep": 0.50,
            "target_timestep": 0.50,
            "beta": 0.1,
            "group_name": "Beta0.1_Baseline"
        },
        {
            "checkpoint": Beta05x10x_01b,
            "source_timestep": 0.50,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": Beta04x10x_01b,
            "source_timestep": 0.40,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": Beta03x10x_01b,
            "source_timestep": 0.30,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": Beta02x10x_01b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": Beta00x10x_01b,
            "source_timestep": 0.00,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },

        # beta: 0.5
        {
            "checkpoint": Beta02x10x_05b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.5,
            "group_name": "Beta0.5"
        },

        # beta: 1.0
        {
            "checkpoint": Beta05x05x_1b,
            "source_timestep": 0.50,
            "target_timestep": 0.50,
            "beta": 1.0,
            "group_name": "Beta1.0_Baseline"
        },
        {
            "checkpoint": Beta05x10x_1b,
            "source_timestep": 0.50,
            "target_timestep": 1.00,
            "beta": 1.0,
            "group_name": "Beta1.0"
        },
        {
            "checkpoint": Beta02x10x_1b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 1.0,
            "group_name": "Beta1.0"
        },

        # beta: 2.0
        {
            "checkpoint": Beta02x10x_2b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 2.0,
            "group_name": "Beta2.0"
        },

        # beta: 3.0
        {
            "checkpoint": Beta02x10x_3b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 3.0,
            "group_name": "Beta3.0"
        },

        # beta: 5.0
        {
            "checkpoint": Beta05x05x_5b,
            "source_timestep": 0.50,
            "target_timestep": 0.50,
            "beta": 5.0,
            "group_name": "Beta5.0_Baseline"
        },
        {
            "checkpoint": Beta05x10x_5b,
            "source_timestep": 0.50,
            "target_timestep": 1.00,
            "beta": 5.0,
            "group_name": "Beta5.0"
        },
        {
            "checkpoint": Beta02x10x_5b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 5.0,
            "group_name": "Beta5.0"
        },
    ]

    for config in model_configs:
        checkpoint = config["checkpoint"]
        source_timestep = config["source_timestep"]
        target_timestep = config["target_timestep"]
        beta = config["beta"]
        group_name = config["group_name"]

        baseline = (source_timestep == target_timestep)
        project_name = "BetaVAE_Quantitative_Eval_Baseline" if baseline else "BetaVAE_Quantitative_Eval"
        model_name = f"{group_name}_VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"

        print(f"\n{'='*100}\nStarting evaluation: {model_name}\n{'='*100}\n")

        run_data_collection_and_evaluation(
            checkpoint=checkpoint,
            data_path=validation_data_path if group == "validation" else test_data_path,
            project_name=project_name,
            model_name=model_name,
            group=group,
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            beta=beta,
            samples_per_class=samples_per_class,
            num_pairs=num_pairs,
            num_interpolations=num_interpolations,
            cfg_scales=[cfg_scale],
            ccfg_scales=[ccfg_scale],
            batch_size=batch_size,
            dataset_name=dataset_name,
            root_path=root_path,
            max_samples=max_samples,
            num_steps=num_steps,
            num_classes=num_classes,
            plot_every_n_batches=plot_every_n_batches,
        )

        print(f"\nCompleted: {model_name}\n{'='*100}\n")
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[INFO] Evaluation script completed successfully!")



# CUDA_VISIBLE_DEVICES=2 python ...