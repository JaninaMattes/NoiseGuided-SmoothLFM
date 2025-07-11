
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics


import os, sys
import gc

from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple


from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm



# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
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

from ldm.trainer_rf_vae import TrainerModuleLatentFlow
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

from ldm.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor # denorm tensor -- just for plotting



torch.set_float32_matmul_precision('high')




#########################################################
#                    Metric Tracker Classes             #
#########################################################
class SmoothnessMetricsTracker(nn.Module):
    """
    Calculates an adapted smoothness metrics PPL (Perceptual Path Length) and ISTD (Interpolation Smoothness STD).
    Based on StyleGAN (Karras et al.) and Smooth Diffusion.
    
    Based on:
    [0] PPL: "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
    [1] Smooth Diffusion: "Crafting Smooth Latent Spaces in Diffusion Models" (Guo et al., 2024)
    """
    def __init__(self, device=None, normalize_step=True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.lpips.eval()
        self.normalize_step = normalize_step
        self.reset()

    def reset(self):
        self.ppls = []
        self.istds = []

    @torch.no_grad()
    def update(self, interpolated_imgs_batch):
        assert interpolated_imgs_batch.dim() == 5, f"Expected 5D tensor, got {interpolated_imgs_batch.dim()}D"
        B, T, C, H, W = interpolated_imgs_batch.shape
        assert T > 1, "Each sequence must contain at least 2 images."

        # Normalize each sequence to [-1, 1] if needed
        min_val, max_val = interpolated_imgs_batch.min(), interpolated_imgs_batch.max()
        if min_val < -1.01 or max_val > 1.01:
            min_per_seq = interpolated_imgs_batch.amin(dim=(2, 3, 4), keepdim=True)
            max_per_seq = interpolated_imgs_batch.amax(dim=(2, 3, 4), keepdim=True)
            denom = (max_per_seq - min_per_seq).clamp(min=1e-5)
            interpolated_imgs_batch = 2 * (interpolated_imgs_batch - min_per_seq) / denom - 1

        batch = interpolated_imgs_batch.to(self.device)
        epsilon = 1.0 / (T - 1) if self.normalize_step else 1.0

        for i in range(B):
            sequence = batch[i]
            dists = []

            for t in range(T - 1):
                d = self.lpips(sequence[t].unsqueeze(0), sequence[t + 1].unsqueeze(0)).item()
                dists.append((d ** 2) / (epsilon ** 2))

            if not dists:
                print("[WARN] No valid LPIPS distances computed.")
                continue

            self.ppls.append(np.mean(dists))
            self.istds.append(np.std(dists))

        print(f"[INFO] Processed {B} sequences for smoothness metrics.")

    @torch.no_grad()
    def aggregate(self):
        if not self.ppls:
            print("Warning: No data in tracker. Call update() before aggregate().")
            return {'ppl': float('nan'), 'istd': float('nan')}

        mean_ppl = np.mean(self.ppls)
        mean_istd = np.mean(self.istds)

        return {'ppl': mean_ppl, 'istd': mean_istd}





def test_smoothness_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = SmoothnessMetricsTracker(device=device, normalize_step=True)

    B, T, C, H, W = 64, 8, 3, 256, 256

    # -----------------------------
    # Smooth interpolations (linear fades)
    # -----------------------------
    smooth_batch = []
    for _ in range(B):
        start_img = torch.rand(1, C, H, W)
        end_img = torch.rand(1, C, H, W)
        seq = torch.cat([
            start_img * (1 - alpha) + end_img * alpha
            for alpha in torch.linspace(0, 1, T)
        ], dim=0)
        smooth_batch.append(seq.unsqueeze(0))  # Add batch dimension

    smooth_batch = torch.cat(smooth_batch, dim=0)  # Shape: (B, T, C, H, W)

    tracker.update(smooth_batch)
    smooth_results = tracker.aggregate()
    print(f"Smooth interpolation results: PPL={smooth_results['ppl']:.4f}, ISTD={smooth_results['istd']:.4f}")

    tracker.reset()

    # -----------------------------
    # Jagged interpolations (random jumps)
    # -----------------------------
    jagged_batch = []
    for _ in range(B):
        seq = torch.rand(T, C, H, W)
        jagged_batch.append(seq.unsqueeze(0))

    jagged_batch = torch.cat(jagged_batch, dim=0)  # Shape: (B, T, C, H, W)

    tracker.update(jagged_batch)
    jagged_results = tracker.aggregate()
    print(f"Jagged interpolation results: PPL={jagged_results['ppl']:.4f}, ISTD={jagged_results['istd']:.4f}")




# ------------------------------------------------
# Test runner
# ------------------------------------------------
if __name__ == "__main__":
    print("Running improved Precision-Recall tests...")
    test_smoothness_metrics()