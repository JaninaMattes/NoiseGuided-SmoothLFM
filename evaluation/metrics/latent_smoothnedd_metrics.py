
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py


import os, sys
import gc

from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib import rcParams

from elatentlpips import ELatentLPIPS


# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from jutils import denorm
from jutils import ims_to_grid
from jutils.vision import tensor2im
from jutils import exists, freeze, default
from jutils import tensor2im, ims_to_grid



# Setup project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
sys.path.append(project_root)

from ldm.trainer_rf_vae import TrainerModuleLatentFlow
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

from ldm.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_tensor, denorm_metrics_tensor



torch.set_float32_matmul_precision('high')






#########################################################
#                    Metric Tracker Classes             #
#########################################################
class LatentSmoothnessMetricsTracker(nn.Module):
    """
    Combines two metrics to evaluate Smoothness in latent space:
    - PPL (Perceptual Path Length): Measures the average perceptual distance between consecutive images in a sequence.
    - ISTD (Interpolation Smoothness STD): Measures the standard deviation of the perceptual distances, indicating how smooth the interpolation is.
    
    
    Based on:
    [0] PPL: "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
    [1] Smooth Diffusion: "Crafting Smooth Latent Spaces in Diffusion Models" (Guo et al., 2024)
    """
    def __init__(self, device=None, normalize_step=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.e_lpips = ELatentLPIPS(encoder="sd15", augment="bg").to(self.device).eval()
        self.normalize_step = normalize_step
        
        self.reset()

    def reset(self):
        self.ppl_values = []
        self.istd_values = []
        self.lpips_values = []


    @torch.no_grad()
    def update(self, sequences):
        """
        sequences: tensor (B, T, C, H, W), normalized to [-1, 1]
        """
        B, T, C, H, W = sequences.shape
        epsilon_jitter = 1e-4
        epsilon = 1.0 / (T - 1) if self.normalize_step else 1.0

        # Normalize for LPIPS (expects [-1, 1])
        sequences = denorm_metrics_tensor(sequences, target_range=(0, 1), dtype='float').to(self.device)

        for i in range(B):
            seq = sequences[i]  # shape: (T, C, H, W)
            dists = []
            for _ in range(T - 1):
                # Sample random t in [0, 1 - ε]
                t = torch.rand(1).item() * (1 - epsilon_jitter)

                # Convert t into fractional index between frames
                idx = t * (T - 1)
                lower = int(torch.floor(torch.tensor(idx)).item())
                upper = min(lower + 1, T - 1)
                alpha = idx - lower

                # Linear interpolation between two frames
                frame0 = lerp(alpha, seq[lower], seq[upper])
                frame1 = lerp(alpha + epsilon_jitter, seq[lower], seq[upper])

                # Ensure correct shape (1, C, H, W)
                x0 = frame0.unsqueeze(0).to(self.device)
                x1 = frame1.unsqueeze(0).to(self.device)

                # Convert from [0,1] → [-1,1] for LPIPS
                d = self.e_lpips(x0 * 2 - 1, x1 * 2 - 1).detach().cpu().item()
                dists.append(d / epsilon)

            if len(dists) > 0:
                self.ppl_values.append(np.sqrt(np.mean(np.square(dists))))
                self.istd_values.append(np.std(dists))
                self.lpips_values.append(np.mean(dists))

        if len(self.ppl_values) == 0:
            print("Warning: No distances computed. Check input sequences.")
            
        # Clean up memory
        del sequences
        torch.cuda.empty_cache()
            

    @torch.no_grad()
    def aggregate(self):
        if len(self.ppl_values) == 0:
            return {"ppl": float("nan"), "istd": float("nan")}
        return {
            "ppl": np.mean(self.ppl_values),
            "istd": np.mean(self.istd_values),
            "lpips": np.mean(self.lpips_values)
        }





############################################
# Latent Image metrics tracker
############################################
class LatentImageMetricsTracker(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()
        self.e_lpips = ELatentLPIPS(encoder="sd15", augment="bg").to(self.device).eval()
        self.e_lpips.eval()

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




# ========== Test Case ========== #
if __name__ == "__main__":

    print("PyTorch CUDA version:", torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Testing LatentSmoothnessTracker ===")
    tracker = LatentSmoothnessTracker(device=device)

    # ----------- Smooth transitions (flat) -----------
    B, T, D = 32, 6, 1024
    base_latent = torch.randn(B, 1, D)
    noise = torch.randn(B, T, D) * 0.05  # small noise
    latents_smooth = base_latent + torch.linspace(0, 1, T).view(1, T, 1) * noise

    tracker.reset()
    tracker.update(latents_smooth)
    smooth_metrics = tracker.aggregate()
    print("Smooth metrics:", smooth_metrics)

    # ----------- Jagged transitions (flat) -----------
    latents_jagged = torch.randn(B, T, D)  # totally random jumps

    tracker.reset()
    tracker.update(latents_jagged)
    jagged_metrics = tracker.aggregate()
    print("Jagged metrics:", jagged_metrics)

    # Assertions with correct keys
    assert smooth_metrics["latent_mdpl"] < jagged_metrics["latent_mdpl"], "latent_mdpl should be lower for smooth transitions"
    assert smooth_metrics["latent_mistd"] < jagged_metrics["latent_mistd"], "latent_mistd should be lower for smooth transitions"
    assert smooth_metrics["latent_cdpl"] < jagged_metrics["latent_cdpl"], "latent_cdpl should be lower for smooth transitions"
    assert smooth_metrics["latent_cistd"] < jagged_metrics["latent_cistd"], "latent_cistd should be lower for smooth transitions"

    print("LatentSmoothnessTracker test passed\n")

    # ========== Test Heavy distorted Paths ========== #
    print("=== Testing Heavy Distorted Paths ===")
    tracker.reset()

    base_latent = torch.randn(B, 1, D)
    noise = torch.randn(B, T, D) * 0.5  # heavy distortion
    latents_heavy_distorted = base_latent + torch.linspace(0, 1, T).view(1, T, 1) * noise
    
    tracker.update(latents_heavy_distorted)
    heavy_distorted_metrics = tracker.aggregate()
    print("Heavy Distorted metrics:", heavy_distorted_metrics)

    # Assertions
    assert heavy_distorted_metrics["latent_mdpl"] > smooth_metrics["latent_mdpl"], "latent_mdpl should be higher for heavy distorted paths"
    assert heavy_distorted_metrics["latent_mistd"] > smooth_metrics["latent_mistd"], "latent_mistd should be higher for heavy distorted paths"
    assert heavy_distorted_metrics["latent_cdpl"] > smooth_metrics["latent_cdpl"], "latent_cdpl should be higher for heavy distorted paths"
    assert heavy_distorted_metrics["latent_cistd"] > smooth_metrics["latent_cistd"], "latent_cistd should be higher for heavy distorted paths"

    print("LatentSmoothnessTracker heavy distortion test passed\n")
    
    # print("=== Testing ImageMetricsTracker ===")
    # tracker = ImageMetricsTracker(num_crops=4, crop_size=128, device=device)
    # tracker.reset()

    # # Example: random images
    # lat_clean = torch.rand(128, 3, 256, 256, device=device)
    # lat_noisy = lat_clean + 0.05 * torch.randn_like(lat_clean, device=device)

    # tracker.update(lat_clean, lat_noisy)
    # print(tracker.aggregate())



    # # ========== Test ImageMetricsTracker ========== #
    # print("=== Testing ImageMetricsTracker ===")
    # tracker = ImageMetricsTracker(num_crops=4, crop_size=128, device=device)
    # tracker.reset()
    # batch_size = 128
    
    # for batch_idx in range(20):
    #     # Simulate batch: random images
    #     lat_clean = torch.rand(batch_size, 3, 256, 256, device=device)
    #     lat_noisy = lat_clean + 0.05 * torch.randn_like(lat_clean, device=device)

    #     tracker.update(lat_clean, lat_noisy)
    #     print(f"Batch {batch_idx + 1}/20 processed.")
        
    # # Aggregate metrics
    # metrics = tracker.aggregate()
    # # Average
    # print("\n=== Final Aggregated Metrics ===")
    # print(f"Global FID: {metrics['gfid']:.6f}")
    # print(f"Local FID : {metrics['lfid']:.6f}" if metrics['lfid'] is not None else "Local FID: N/A")
    # print(f"SSIM      : {metrics['ssim']:.6f}")     
    # print(f"PSNR      : {metrics['psnr']:.6f}")
    # print(f"MSE       : {metrics['mse']:.6f}")
    # print(f"MAE       : {metrics['mae']:.6f}")
    # print(f"LPIPS     : {metrics['lpips']:.6f}")
    
    
    
    # ========== Test ImageMetricsTracker ========== #

    # print("=== Testing ImageMetricsTracker ===")
    # tracker = ImageMetricsTracker(num_crops=4, crop_size=128, device=device)
    # gfid, lfid, ssim, psnr, mse, mae, lpips = [], [], [], [], [], [], []
    
    # batch_size = 128
    
    # for batch_idx in range(20):
    #     # Simulate batch: random images
    #     tracker.reset()
        
    #     lat_clean = torch.rand(128, 3, 256, 256, device=device)
    #     lat_noisy = lat_clean + 0.05 * torch.randn_like(lat_clean, device=device)

    #     tracker.update(lat_clean, lat_noisy)
    #     print(f"Batch {batch_idx + 1}/20 processed.")
        
    #     # Aggregate metrics
    #     metrics = tracker.aggregate()
    #     # print(f"→ Batch {batch_idx + 1} Metrics: {metrics}")
    #     gfid.append(metrics['gfid'])
    #     lfid.append(metrics['lfid'] if metrics['lfid'] is not None else float('nan'))
    #     ssim.append(metrics['ssim'])
    #     psnr.append(metrics['psnr'])
    #     mse.append(metrics['mse'])
    #     mae.append(metrics['mae'])
    #     lpips.append(metrics['lpips'])
        
    #     # Optional: free VRAM
    #     torch.cuda.empty_cache()

    # # Average
    # print("\n=== Final Aggregated Metrics ===")
    # print(f"Global FID: {torch.tensor(gfid).mean().item():.6f}")
    # print(f"Local FID : {torch.tensor(lfid).mean().item():.6f}")
    # print(f"SSIM      : {torch.tensor(ssim).mean().item():.6f}")
    # print(f"PSNR      : {torch.tensor(psnr).mean().item():.6f}")
    # print(f"MSE       : {torch.tensor(mse).mean().item():.6f}")
    # print(f"MAE       : {torch.tensor(mae).mean().item():.6f}")
    # print(f"LPIPS     : {torch.tensor(lpips).mean().item():.6f}")

    
    # # ========== Test LatentSimilarityTracker ========== #
    # print("=== Testing LatentSimilarityTracker ===")
    # tracker = LatentSimilarityTracker()

    # # Case 1: latent vectors
    # lat_clean = torch.rand(10, 4, 32, 32)
    # lat_noisy = lat_clean + 0.05 * torch.randn_like(lat_clean)
    # tracker.update(lat_clean, lat_noisy)

    # # Case 2: image-style decoded outputs (in [0, 1])
    # img_clean = torch.rand(10, 4, 32, 32)
    # img_noisy = img_clean + 0.05 * torch.randn_like(img_clean)
    # tracker.update(img_clean, img_noisy)

    # print(tracker.aggregate())
    

    # # ========== Test LatentSmoothnessTracker ========== #
    # print("\n=== Testing LatentSmoothnessTracker ===")
    # latent_smoothness_tracker = LatentSmoothnessTracker(device=device)

    # # Simulate a batch of 4 latent sequences, each with 16 interpolation steps
    # B, T, D, H, W = 10, 16, 4, 32, 32
    # latent_sequences = torch.randn(B, T, D, H, W) * 0.1  # Small noise = smooth interpolation
    # latent_sequences = latent_sequences.clamp(-1, 1)

    # # Update tracker with dummy latent sequences
    # latent_smoothness_tracker.update(latent_sequences)

    # # Print results
    # results = latent_smoothness_tracker.aggregate()
    # print("→ Latent PPL :", f"{results['ppl']:.6f}")
    # print("→ Latent ISTD:", f"{results['istd']:.6f}")


    
    # # ========== Test SmoothnessMetricsTracker ========== #
    # print("\n=== Testing SmoothnessMetricsTracker ===")
    
    # tracker = SmoothnessMetricsTracker(device=device)

    # # Dummy data: 2 sequences, each with 10 images (T), 3 channels, 256x256, in [-1, 1]
    # B, T, C, H, W = 2, 10, 3, 256, 256
    # dummy_data = torch.randn(B, T, C, H, W) * 0.1  # Small noise → for a smooth sequence
    # dummy_data = dummy_data.clamp(-1, 1)

    # tracker.update(dummy_data)

    # metrics = tracker.aggregate()
    # print("\n=== Final Aggregated Metrics ===")
    # print(f"PPL  : {metrics['ppl']:.6f}")
    # print(f"ISTD : {metrics['istd']:.6f}")



    # # Dummy data: 2 sequences, each with 10 images (T), 3 channels, 64x64, in [-1, 1]
    # B, T, C, H, W = 10, 10, 3, 256, 256
    # dummy_data = torch.randn(B, T, C, H, W) * 0.1  # Small noise → for a smooth sequence
    # dummy_data = dummy_data.clamp(-1, 1)

    # tracker.update(dummy_data)

    # metrics = tracker.aggregate()
    # print("\n=== Final Aggregated Metrics ===")
    # print(f"PPL  : {metrics['ppl']:.6f}")
    # print(f"ISTD : {metrics['istd']:.6f}")



    # # Dummy data: 2 sequences, each with 10 images (T), 3 channels, 64x64, in [-1, 1]
    # B, T, C, H, W = 1, 10, 3, 256, 256
    # dummy_data = torch.randn(B, T, C, H, W) * 0.1  # Small noise → for a smooth sequence
    # dummy_data = dummy_data.clamp(-1, 1)

    # tracker.update(dummy_data)

    # metrics = tracker.aggregate()
    # print("\n=== Final Aggregated Metrics ===")
    # print(f"PPL  : {metrics['ppl']:.6f}")
    # print(f"ISTD : {metrics['istd']:.6f}")
