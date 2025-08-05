
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
#                    GIF/ MP4 Generator                 #
#########################################################
def sharpen_image(img_np):
    """img_np: shape (H, W, C), dtype uint8 or float32 in range 0-1 or 0-255"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    sharpened = cv2.filter2D(img_np, -1, kernel)
    return sharpened


def frames2mp4(vpath, frames, fps=10, sharpen=True):
    """
    frames: list of np.array images, shape (H, W, C)
    vpath: output path to mp4 or gif
    """
    import moviepy.editor as mpy

    if sharpen:
        frames = [sharpen_image(f) for f in frames]

    clip = mpy.ImageSequenceClip(frames, fps=fps)

    if vpath.endswith(".gif"):
        clip.write_gif(vpath, fps=fps)
    else:
        clip.write_videofile(vpath, fps=fps)




#########################################################
#               Helper Linear Interpolation             #
#########################################################
def lerp(t, v0, v1):
    return v0 * (1 - t) + v1 * t

def generate_interpolation_sequence(start_img, end_img, num_steps=8):
    seq = torch.stack([lerp(t, start_img, end_img) 
                       for t in torch.linspace(0, 1, num_steps)], dim=0)
    return seq




#########################################################
#                    Metric Tracker Classes             #
#########################################################
class SmoothnessMetricsTracker(nn.Module):
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
        self.lpips = LPIPS(net_type='vgg').to(self.device).eval()
        self.normalize_step = normalize_step
        self.reset()

    def reset(self):
        self.ppl_values = []
        self.istd_values = []
        self.lpips_raw = []

    @torch.no_grad()
    def update(self, sequences):
        """
        Args:
            sequences (Tensor): (B, T, C, H, W) tensor in [-1, 1]
        """
        B, T, C, H, W = sequences.shape
        epsilon = 1.0 / (T - 1) if self.normalize_step else 1.0

        # Convert to [0, 1] for consistency with LPIPS expectations
        sequences = denorm_metrics_tensor(sequences, target_range=(0, 1), dtype='float').to(self.device)

        for b in range(B):
            seq = sequences[b]  # (T, C, H, W)
            dists = []

            for t in range(T - 1):
                x0 = seq[t].unsqueeze(0)  # (1, C, H, W)
                x1 = seq[t + 1].unsqueeze(0)  # (1, C, H, W)

                # Convert to [-1, 1] for LPIPS
                d = self.lpips(x0 * 2 - 1, x1 * 2 - 1).item()
                dists.append(d / epsilon)
                self.lpips_raw.append(d)

            if len(dists) > 0:
                self.ppl_values.append(np.mean(dists))
                self.istd_values.append(np.std(dists))

        torch.cuda.empty_cache()

    @torch.no_grad()
    def aggregate(self):
        if len(self.ppl_values) == 0:
            return {
                "ppl": float("nan"),
                "istd": float("nan"),
                "lpips_mean": float("nan"),
                "lpips_std": float("nan"),
            }

        return {
            "ppl": np.mean(self.ppl_values),
            "istd": np.mean(self.istd_values),
            "lpips_mean": np.mean(self.lpips_raw),
            "lpips_std": np.std(self.lpips_raw),
        }






def test_smoothness_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = SmoothnessMetricsTracker(device=device, normalize_step=True)

    B, T, C, H, W = 64, 8, 3, 64, 64  # smaller size for fast testing

    print("===================================")
    print("Test 1: Smooth Interpolations")
    print("===================================")
    start = torch.rand(B, C, H, W)
    end = torch.rand(B, C, H, W)

    # Generate interpolated sequences
    alphas = torch.linspace(0, 1, T).to(device)
    smooth_batch = torch.stack([
        torch.stack([lerp(a, start[b].to(device), end[b].to(device)) for a in alphas], dim=0)
        for b in range(B)
    ], dim=0)  # (B, T, C, H, W)

    tracker.update(smooth_batch)
    results = tracker.aggregate()
    print(f"Smooth Interp - PPL: {results['ppl']:.6f}, ISTD: {results['istd']:.6f}, LPIPS: {results['lpips_mean']:.6f} ± {results['lpips_std']:.6f}")
    tracker.reset()

    print("\n===================================")
    print("Test 2: Jagged Sequences")
    print("===================================")
    jagged_batch = torch.rand(B, T, C, H, W)
    tracker.update(jagged_batch.to(device))
    results = tracker.aggregate()
    print(f"Jagged - PPL: {results['ppl']:.6f}, ISTD: {results['istd']:.6f}, LPIPS: {results['lpips_mean']:.6f} ± {results['lpips_std']:.6f}")
    tracker.reset()

    print("\n===================================")
    print("Test 3: Identical Frames")
    print("===================================")
    identical = torch.rand(B, C, H, W)
    identical_batch = identical.unsqueeze(1).repeat(1, T, 1, 1, 1)
    tracker.update(identical_batch.to(device))
    results = tracker.aggregate()
    print(f"Identical - PPL: {results['ppl']:.6f}, ISTD: {results['istd']:.6f}, LPIPS: {results['lpips_mean']:.6f} ± {results['lpips_std']:.6f}")
    tracker.reset()

    # Optional: simple assertions for test sanity
    assert results["ppl"] < 1e-4, "Identical image PPL should be near zero"
    assert results["istd"] < 1e-4, "Identical image ISTD should be near zero"
    assert results["lpips_mean"] < 1e-4, "Identical image LPIPS mean should be near zero"
    assert results["lpips_std"] < 1e-4, "Identical image LPIPS std should be near zero"

    
# ------------------------------------------------
# Test runner
# ------------------------------------------------
if __name__ == "__main__":
    print("Running improved Precision-Recall tests...")
    test_smoothness_metrics()
 
    
    
# CUDA_VISIBLE_DEVICES=0 python ...