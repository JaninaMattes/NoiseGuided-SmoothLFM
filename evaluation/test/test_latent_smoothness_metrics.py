
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

import cv2
import moviepy.editor as mpy


from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm

# Latent LPIPS
from elatentlpips import ELatentLPIPS


# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
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

from ldm.utils.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor # denorm tensor -- just for plotting



torch.set_float32_matmul_precision('high')



#########################################################
#                    GIF/ MP4 Generator                 #
#########################################################
def sharpen_image(img_np, strength=1.0):
    """
    Applies a sharpening filter to an image.
    
    Args:
        img_np: np.array of shape (H, W, C), dtype uint8 or float32
        strength: Controls sharpness intensity (1.0 = default)
    
    Returns:
        Sharpened image as np.uint8
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + strength, -1],
                       [0, -1, 0]])
    
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    return cv2.filter2D(img_np, -1, kernel)

def frames2mp4(vpath, frames, fps=10, sharpen=True, sharpen_strength=1.0):
    """
    Generates an MP4 or GIF from a list of frames.

    Args:
        vpath: Output path (.mp4 or .gif)
        frames: List of np.array frames (H, W, C)
        fps: Frames per second
        sharpen: Whether to apply sharpening filter
        sharpen_strength: Strength of sharpening (default: 1.0)
    """
    if sharpen:
        frames = [sharpen_image(f, strength=sharpen_strength) for f in frames]

    clip = mpy.ImageSequenceClip(frames, fps=fps)

    if vpath.endswith(".gif"):
        clip.write_gif(vpath, fps=fps)
    else:
        clip.write_videofile(vpath, fps=fps, codec="libx264", audio=False, logger=None)

    del clip



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


    @torch.no_grad()
    def update(self, sequences):
        """
        sequences: tensor (B, T, C, H, W), normalized to [-1, 1]
        """
        B, T, C, H, W = sequences.shape
        epsilon = 1.0 / (T - 1) if self.normalize_step else 1.0

        for i in range(B):
            seq = sequences[i]
            dists = []
            for t in range(T - 1):
                x0 = seq[t:t+1].to(self.device)
                x1 = seq[t+1:t+2].to(self.device)
                d = self.e_lpips(x0, x1, normalize=False).detach().cpu().mean().item()
                dists.append((d ** 2) / (epsilon ** 2))

            if len(dists) > 0:
                self.ppl_values.append(np.mean(dists))
                self.istd_values.append(np.std(dists))
                
        # If no distances were computed, warn
        if len(self.ppl_values) == 0:
            print("Warning: No distances computed. Check input sequences.")
            

    @torch.no_grad()
    def aggregate(self):
        if len(self.ppl_values) == 0:
            return {"ppl": float("nan"), "istd": float("nan")}
        return {
            "ppl": np.mean(self.ppl_values),
            "istd": np.mean(self.istd_values)
        }




def test_smoothness_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = LatentSmoothnessMetricsTracker(device=device, normalize_step=True)

    B, T, C, H, W = 64, 10, 4, 32, 32  # - Use a smaller for faster test

    print("===================================")
    print("Test 1: All Smooth Interpolations")
    print("===================================")

    smooth_batch = []
    for _ in range(B):
        start_img = torch.rand(1, C, H, W)
        end_img = torch.rand(1, C, H, W)
        seq = torch.cat([lerp(alpha, start_img, end_img) 
                         for alpha in torch.linspace(0, 1, T)], dim=0)
        smooth_batch.append(seq.unsqueeze(0))
    smooth_batch = torch.cat(smooth_batch, dim=0)  # (B, T, C, H, W)

    tracker.update(smooth_batch)
    smooth_results = tracker.aggregate()
    print(f"Smooth: PPL={smooth_results['ppl']:.6f}, ISTD={smooth_results['istd']:.6f}")
    tracker.reset()

    print("\n===================================")
    print("Test 2: All Jagged Interpolations")
    print("===================================")

    jagged_batch = []
    for _ in range(B):
        seq = torch.rand(T, C, H, W)
        jagged_batch.append(seq.unsqueeze(0))
    jagged_batch = torch.cat(jagged_batch, dim=0)

    tracker.update(jagged_batch)
    jagged_results = tracker.aggregate()
    print(f"Jagged: PPL={jagged_results['ppl']:.6f}, ISTD={jagged_results['istd']:.6f}")
    tracker.reset()

    print("\n===================================")
    print("Test 3: All Identical Images")
    print("===================================")

    identical_batch = []
    for _ in range(B):
        img = torch.rand(1, C, H, W)
        seq = img.repeat(T, 1, 1, 1)
        identical_batch.append(seq.unsqueeze(0))
    identical_batch = torch.cat(identical_batch, dim=0)

    tracker.update(identical_batch)
    identical_results = tracker.aggregate()
    print(f"Identical: PPL={identical_results['ppl']:.6f}, ISTD={identical_results['istd']:.6f}")
    tracker.reset()




# ------------------------------------------------
# Test runner
# ------------------------------------------------
if __name__ == "__main__":
    print("Running improved Precision-Recall tests...")
    test_smoothness_metrics()
 
    
    
# CUDA_VISIBLE_DEVICES=2 python ...