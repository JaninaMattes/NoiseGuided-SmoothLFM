



# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import cv2
import numpy as np

import os, sys
import gc

from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import seed_everything
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

import cv2
from moviepy import ImageSequenceClip


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







############################################
#   Utils
############################################

def clear_folder(path: Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        print(f"[INFO] Cleared existing folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created folder: {path}")


#########################################################
#                    GIF/ MP4 Generator                 #
#########################################################
def unsharp_mask(img, amount=1.0, threshold=0):
    """
    Applies unsharp masking for image sharpening.

    Args:
        img: np.uint8 RGB image
        amount: Sharpening strength (e.g., 1.0)
        threshold: Optional threshold to avoid amplifying noise

    Returns:
        Sharpened image (uint8)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    if threshold > 0:
        # Avoid enhancing low-contrast areas
        diff = np.abs(img.astype(np.int16) - blurred.astype(np.int16))  # int16 to avoid wraparound
        low_contrast_mask = (diff < threshold).all(axis=-1)
        low_contrast_mask = np.stack([low_contrast_mask] * 3, axis=-1)
        np.copyto(sharpened, img, where=low_contrast_mask)

    return sharpened



def enhance_image(img_np, sharpen_strength=1.0, saturation_boost=0.1):
    """
    Enhances image by sharpening and boosting saturation.
    
    Args:
        img_np: np.array of shape (H, W, C), dtype uint8 or float32
        sharpen_strength: Strength of sharpening
        saturation_boost: Percent boost in saturation (e.g., 0.1 = +10%)
    
    Returns:
        Enhanced RGB image (uint8)
    """
    # Convert float to uint8 if needed
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Apply unsharp mask sharpening
    img_sharp = unsharp_mask(img_np, amount=sharpen_strength)

    # Boost saturation via HSV
    img_hsv = cv2.cvtColor(img_sharp, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[..., 1] *= (1.0 + saturation_boost)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)
    img_enhanced = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img_enhanced




def sharpen_filter_image(img_np, strength=1.0):
    """
    Applies a sharpening filter to an image.

    Args:
        img_np: np.ndarray (H, W, C) or (H, W), dtype uint8
        strength: float, sharpening strength (default 1.0)

    Returns:
        np.ndarray: sharpened image (uint8)
    """
    assert img_np.dtype == np.uint8, "Input image must be of type uint8"
    
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(img_np, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)



def sharpen_image(img_np, strength=0.1):
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



def frames2mp4(vpath, frames, fps=10, sharpen=True, sharpen_strength=0.1):
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

    clip = ImageSequenceClip(frames, fps=fps)

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




##########################################################
#               Smoothness Metrics Tracker                #
##########################################################
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
        self.lpips_values = []


    @torch.no_grad()
    def update(self, sequences):
        """
        sequences: tensor (B, T, C, H, W), normalized to [-1, 1]
        """
        B, T, C, H, W = sequences.shape
        if T < 3:
            print(f"[WARN] Sequence too short for smoothness: T={T}")
            return

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
                d = self.lpips(x0 * 2 - 1, x1 * 2 - 1).detach().cpu().item()
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
            return {"ppl": float("nan"), "istd": float("nan"), "lpips_mean": float("nan"), "lpips_std": float("nan")}
        return {
            "ppl": np.mean(self.ppl_values),
            "istd": np.mean(self.istd_values),
            "lpips_mean": np.mean(self.lpips_values),
            "lpips_std": np.std(self.lpips_values),
        }





############################################
# Image metrics tracker
############################################
class ImageMetricsTracker(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()
        self.lpips = LPIPS(net_type='vgg').to(self.device).eval()
        self.lpips.eval()

        self.reset()

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.maes = []
        self.cossims = []
        self.lpips_scores = []
        
        
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
        self.lpips_scores.append(self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1).detach().cpu())

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
            lpips=torch.stack(self.lpips_scores).mean().item(),
            cossim=torch.cat(self.cossims).mean().item()
        )









#########################################################
#           Collect samples from the dataset            #
#########################################################
from collections import defaultdict
from typing import List, Tuple

@torch.no_grad()
def collect_samples(
    data,
    class_labels: List[int],
    source_timestep: float,
    samples_per_class: int = 10,
    group_name='validation'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collects a fixed number of samples per class from a validation dataloader.
    Returns latents, labels, and corresponding images.
    """
    collected_latents = defaultdict(list)
    collected_images = defaultdict(list)
    
    # Dataloader
    if group_name == 'validation':
        val_loader = data.val_dataloader()
    else:
        val_loader = data.test_dataloader()
        
    latent_key = f'latents_{source_timestep:.2f}'

    for batch_idx, batch in enumerate(val_loader):
        latents = batch[latent_key].detach().cpu()
        labels = batch['label'].view(-1).detach().cpu()
        images = batch['image'].detach().cpu()

        for label in class_labels:
            needed = samples_per_class - len(collected_latents[label])
            if needed <= 0:
                continue

            class_mask = labels == label
            selected_latents = latents[class_mask]
            selected_images = images[class_mask]

            if selected_latents.size(0) > 0:
                to_take = min(needed, selected_latents.size(0))
                collected_latents[label].extend(selected_latents[:to_take])
                collected_images[label].extend(selected_images[:to_take])

        if all(len(collected_latents[label]) >= samples_per_class for label in class_labels):
            print(f"[INFO] Collected enough samples for all classes after {batch_idx + 1} batches.")
            break

    all_latents, all_labels, all_images = [], [], []
    for label in class_labels:
        latents_list = collected_latents[label]
        images_list = collected_images[label]

        if len(latents_list) < samples_per_class:
            print(f"[WARNING] Not enough samples for class {label}: got {len(latents_list)}, needed {samples_per_class}. Skipping.")
            continue 


        stacked_latents = torch.stack(latents_list[:samples_per_class], dim=0)
        stacked_images = torch.stack(images_list[:samples_per_class], dim=0)
        label_tensor = torch.full((samples_per_class,), label, dtype=torch.long)

        all_latents.append(stacked_latents)
        all_labels.append(label_tensor)
        all_images.append(stacked_images)

    return (
        torch.cat(all_latents, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_images, dim=0)
    )



def get_dataloader_by_group(data_module, group: str):
    if group == "validation":
        return data_module.val_dataloader()
    elif group == "test":
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Unsupported group: {group}")
    
    
    
    
#########################################################
#               Linear Interpolation Grid                #
#########################################################
@torch.no_grad()
def interpolate_vectors(z1, z2, alpha_vals, mode="linear", dot_threshold=0.9995):
    """
    Interpolates between z1 and z2 using either 'linear' or 'slerp' mode.
    Returns a tensor of shape (B, D) where B = len(alpha_vals).
    """
    if mode == "linear":
        return torch.stack([
            (1 - alpha) * z1 + alpha * z2 for alpha in alpha_vals
        ])
    elif mode == "slerp":
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        dot = torch.dot(z1_norm, z2_norm).clamp(-1.0, 1.0)

        if torch.abs(dot) > dot_threshold:
            return torch.stack([
                torch.lerp(z1, z2, alpha) for alpha in alpha_vals
            ])
        else:
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            return torch.stack([
                torch.sin((1.0 - alpha) * omega) / sin_omega * z1_norm +
                torch.sin(alpha * omega) / sin_omega * z2_norm
                for alpha in alpha_vals
            ])
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")



def is_valid_tensor(tensor, min_size):
    return tensor is not None and tensor.size(0) >= min_size



@torch.no_grad()
def linear_interpolation_grid_with_evaluation(
    cls1_latents, 
    cls2_latents,
    cls1_images, 
    cls2_images,
    fm_module,
    results_dir,
    num_pairs=10,
    num_interpolations=24,
    cfg_scale=3.0,
    ccfg_scale=1.0,
    tracker=None,
    image_tracker=None,
    create_video=False,
    create_grid=False,
    sharpen=False,
    sharpen_strength=0.3,
    saturation_boost=0.2,
    video_fps=8,
    upscale_to=256,
    filename_suffix="interpolation",
    interp_type="linear",
    use_labels=False,
    num_classes=1000,
    device=None
):
    # Early exit if inputs are empty or too small
    if not all([
        is_valid_tensor(cls1_latents, num_pairs),
        is_valid_tensor(cls2_latents, num_pairs),
        is_valid_tensor(cls1_images, num_pairs),
        is_valid_tensor(cls2_images, num_pairs)
    ]):
        print(f"[WARNING] Skipping interpolation: Not enough data for {num_pairs} pairs.")
        return None

    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    alpha_lin_space = torch.linspace(0, 1, num_interpolations)
    generated_rows = []

    print(f"[INFO] Interpolating {num_pairs} pairs, {num_interpolations} steps each...")

    for i in tqdm(range(num_pairs), desc="Interpolating pairs"):
        x1 = cls1_latents[i].unsqueeze(0)
        x2 = cls2_latents[i].unsqueeze(0)

        # Encode third stage
        context_z1 = fm_module.encode_third_stage(x1).to(torch.float32).squeeze(0)
        context_z2 = fm_module.encode_third_stage(x2).to(torch.float32).squeeze(0)

        # Interpolate context
        interp_context = interpolate_vectors(context_z1, context_z2, alpha_lin_space, mode=interp_type).to(device)

        B = interp_context.shape[0]
        C, H, W = x1.size()[1:]
        
        # Generate a single, fixed noise for this interpolation trajectory
        noise_seed = torch.randn(C, H, W, device=device)
        noise_x = noise_seed.unsqueeze(0).repeat(B, 1, 1, 1)

        # Handle label logic
        if use_labels:
            half = num_interpolations // 2
            row_labels = torch.cat([
                torch.full((half,), 0, dtype=torch.long),  # Replace 0 with cls1 label if needed
                torch.full((num_interpolations - half,), 1, dtype=torch.long)  # Replace 1 with cls2 label if needed
            ])
            labels = row_labels.to(device)
            print(f"[INFO] Pair {i}: Using dummy labels for interpolation (adjust if needed).")
        else:
            labels = None

        # Universal conditional context
        uc_cond_context = torch.zeros_like(interp_context)

        uc_cond = (
            torch.full((interp_context.size(0),), num_classes, device=device, dtype=torch.long)
            if labels is not None else None
        )

        sample_kwargs = {
            "num_steps": 50,
            "progress": False,
            "cfg_scale": cfg_scale,
            "ccfg_scale": ccfg_scale,
            "context": interp_context,
            "uc_cond_context": uc_cond_context,
            "y": labels,
            "uc_cond": uc_cond
        }

        samples = fm_module.model.generate(x=noise_x, **sample_kwargs)
        samples = fm_module.decode_first_stage(samples)
        
        # Update tracker if provided
        if tracker is not None:
            tracker.update(samples.unsqueeze(0)) # 1, T, C, H, W
            print(f"[INFO] Updated tracker for pair {i} with {samples.shape[0]} samples.")

        if image_tracker is not None:        
            pred_pair = samples[[0, -1]]  # first and last generated frames
            gt_pair = torch.stack([cls1_images[i], cls2_images[i]])

            print(f"[INFO] Shape pred_pair: {pred_pair.shape}, gt_pair: {gt_pair.shape}")
            assert pred_pair.shape == gt_pair.shape, f"Shape mismatch: {pred_pair.shape} vs {gt_pair.shape}"
            image_tracker.update(target=gt_pair, pred=pred_pair)
            print(f"[INFO] Updated image tracker for pair {i} with {pred_pair.shape[0]} samples.")
            
            
        # Convert to denormalized images for visualization
        row_images = denorm_tensor(samples).detach().cpu() # converted to [0, 255]
        row_images = torch.stack([FT.resize(im, [upscale_to, upscale_to]) for im in row_images])

        start_img = FT.resize(denorm_tensor(cls1_images[i].unsqueeze(0))[0].cpu(), [upscale_to, upscale_to])
        end_img = FT.resize(denorm_tensor(cls2_images[i].unsqueeze(0))[0].cpu(), [upscale_to, upscale_to])
        full_row = torch.cat([start_img.unsqueeze(0), row_images, end_img.unsqueeze(0)], dim=0)

        generated_rows.append(full_row)

        if create_video:
            frames = [im.detach().cpu().permute(1, 2, 0).numpy() for im in full_row]
            video_path = os.path.join(results_dir, f"pair_{i:02d}.{'gif' if video_fps < 10 else 'mp4'}")
            frames2mp4(video_path, frames, fps=video_fps, sharpen=sharpen, sharpen_strength=sharpen_strength, saturation_boost=saturation_boost)
            print(f"[INFO] Saved video: {video_path}")

        # Cleanup
        del x1, x2, context_z1, context_z2, interp_context, noise_x, samples, row_images, full_row
        torch.cuda.empty_cache()

    if create_grid:
        all_rows = torch.cat(generated_rows, dim=0)
        nrow = num_interpolations + 2
        grid = torchvision.utils.make_grid(all_rows, nrow=nrow, padding=2)
        grid_np = grid.permute(1, 2, 0).numpy()

        rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

        fig_width = grid_np.shape[1] / 100
        fig_height = grid_np.shape[0] / 100

        # Limit figure size
        max_fig_size = 20
        fig_width = min(fig_width, max_fig_size)
        fig_height = min(fig_height, max_fig_size)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(grid_np)
        ax.axis('off')
        plt.title("Interpolation Grid with Anchors")
        plt.tight_layout()
        grid_path = os.path.join(results_dir, f"grid_{filename_suffix}.png")
        plt.savefig(grid_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved grid image: {grid_path}")




import pandas as pd
import csv

@torch.no_grad()
def run_interpolation_and_evaluation(
    checkpoint,
    data_path,
    interpolation_dict,
    project_name,
    model_name,
    group,
    source_timestep,
    target_timestep,
    beta,
    samples_per_class=14,
    num_interpolations=20,
    num_pairs=12,
    cfg_scale=3.0,
    ccfg_scale=1.0,
    batch_size=24,
    selected_video_pairs=None,
    sharpen=False,
    sharpen_strength=0.1,
    results_root="./results",
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    fm_module = TrainerModuleLatentFlow.load_from_checkpoint(checkpoint, map_location='cpu')
    fm_module.eval()
    freeze(fm_module.model)
    fm_module.to(device)

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

    # Setup results directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_results_dir = Path(results_root) / project_name / 'Interpolations' / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples
    print("[INFO] Task (0): Collecting samples ...")
    all_classes = {cls for pair in interpolation_dict.values() for cls in pair}
    latents, labels, images = collect_samples(
        data=data,
        class_labels=list(all_classes),
        source_timestep=source_timestep,
        samples_per_class=samples_per_class,
        group_name=group
    )
    
    if latents is None or labels is None or images is None or latents.size(0) == 0:
        print("[ERROR] No samples collected. Exiting early.")
        return

    
    # Prepare CSV metrics log
    metrics_records = []

    # Perform interpolations
    print("[INFO] Task (1): Interpolating pairs ...")

    
    for interp_name, (cls_a, cls_b) in interpolation_dict.items():
        try:
            cls1_mask = labels == cls_a
            cls2_mask = labels == cls_b

            latents_1 = latents[cls1_mask][:num_pairs].to(device)
            latents_2 = latents[cls2_mask][:num_pairs].to(device)
            imgs_1 = images[cls1_mask][:num_pairs].to(device)
            imgs_2 = images[cls2_mask][:num_pairs].to(device)

            # Skip if either class has fewer samples than num_pairs
            if latents_1.size(0) < num_pairs or latents_2.size(0) < num_pairs:
                print(f"[WARNING] Skipping pair {interp_name} due to insufficient samples ({latents_1.size(0)}, {latents_2.size(0)})")
                continue
            
            
            interp_dir = base_results_dir / interp_name
            interp_dir.mkdir(exist_ok=True)

            print(f"[INFO] Running interpolation: {interp_name} ({cls_a} → {cls_b})")

            tracker = SmoothnessMetricsTracker(device=device)
            image_tracker = ImageMetricsTracker(device=device)
            create_video = False if not selected_video_pairs else interp_name in selected_video_pairs

            linear_interpolation_grid_with_evaluation(
                cls1_latents=latents_1,
                cls2_latents=latents_2,
                cls1_images=imgs_1,
                cls2_images=imgs_2,
                fm_module=fm_module,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                results_dir=str(interp_dir),
                num_pairs=num_pairs,
                num_interpolations=num_interpolations,
                tracker=tracker,
                image_tracker=image_tracker, 
                create_video=create_video,
                create_grid=True,
                sharpen=sharpen,
                sharpen_strength=sharpen_strength,
                video_fps=8,
                upscale_to=256,
                filename_suffix=f"{cls_a}_{cls_b}",
                interp_type="linear",
                device=device,
                plot_samples=True,  # Set to True to plot samples
                use_labels=False,  # Set to False to avoid using labels in interpolation
            )
        except Exception as e:
            print(f"[ERROR] Skipping pair '{interp_name}' due to error: {e}")
            continue

        # Update trackers
        tracker_metrics = tracker.aggregate()
        image_metrics = image_tracker.aggregate()

        print(f"[METRICS] Average PPL: {tracker_metrics['ppl']:.8f}, ISTD: {tracker_metrics['istd']:.8f}")
        print(f"[METRICS] SSIM: {image_metrics['ssim']:.8f}, PSNR: {image_metrics['psnr']:.8f}, MSE: {image_metrics['mse']:.8f}, "
                  f"MAE: {image_metrics['mae']:.8f}, LPIPS: {image_metrics['lpips']:.8f}, COSSIM: {image_metrics['cossim']:.8f}")

        metrics_records.append({
                "pair_name": interp_name,
                "cls_a": cls_a,
                "cls_b": cls_b,
                "cfg": cfg_scale,
                "ccfg": ccfg_scale,
                "cossim": image_metrics["cossim"],
                "ssim": image_metrics["ssim"],
                "psnr": image_metrics["psnr"],
                "mse": image_metrics["mse"],
                "mae": image_metrics["mae"],
                "lpips": image_metrics.get("lpips", float("nan")),
                "ppl": tracker_metrics.get("ppl", float("nan")),
                "istd": tracker_metrics.get("istd", float("nan")),
                "lpips_mean": image_metrics.get("lpips_mean", float("nan")),
                "lpips_std": image_metrics.get("lpips_std", float("nan")),
            })
        
        print(f"[INFO] Metrics for {interp_name}: {metrics_records[-1]}")
        print(f"[INFO] Completed interpolation for {interp_name} ({cls_a} → {cls_b})")
        print("=" * 60)

        # print metrics summary
        print(f"Pair: {interp_name} ({cls_a} → {cls_b})")
        for k, v in metrics_records[-1].items():
            if k not in {"pair_name", "cls_a", "cls_b"}:
                print(f"{k.upper():<8}: {v:.8f}")
        print("=" * 60)

            
        # Save partial results to CSV
        partial_df = pd.DataFrame(metrics_records)
        partial_df.to_csv(base_results_dir / f"interpolation_metrics_partial_{model_name}.csv", index=False)
        print(f"[INFO] Saved metrics for {interp_name} to CSV.")

        tracker.reset()
        image_tracker.reset()
        
        del latents_1, latents_2, imgs_1, imgs_2, tracker, image_tracker
        torch.cuda.empty_cache()
        gc.collect()
        
    # Finalize metrics collection
    print("[INFO] Task (1): Completed all interpolations.")
    
    
    # Compute overall averages
    df = pd.DataFrame(metrics_records)
    summary_row = {
        "pair_name": "Overall_Average",
        "cls_a": -1,
        "cls_b": -1,
        "ssim": df["ssim"].mean(),
        "psnr": df["psnr"].mean(),
        "mse": df["mse"].mean(),
        "mae": df["mae"].mean(),
        "lpips": df["lpips"].mean(),
        "cossim": df["cossim"].mean(),
        "ppl": df["ppl"].mean(),
        "istd": df["istd"].mean(),
        "lpips_mean_interpolation": df["lpips"].mean() if "lpips_mean" in df.columns else float("nan"),   # from smoothness tracker
        "lpips_std_interpolation": df["lpips"].std() if "lpips_std" in df.columns else float("nan"),      # from smoothness tracker
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # Print summary
    print("\n" + "=" * 60)
    print("Overall Interpolation Metrics Summary:")
    for k, v in summary_row.items():
        if k not in {"pair_name", "cls_a", "cls_b"}:
            print(f"Average {k.upper():<6}: {v:.6f}")
    print("=" * 60 + "\n")

    # Save CSV
    csv_path = base_results_dir / f"interpolation_metrics_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics CSV to: {csv_path}")
    print("[INFO] Evaluation complete!")
    
    # Clear memory
    del df, summary_row
    torch.cuda.empty_cache()
    gc.collect()
    
    
    
    
    




if __name__ == "__main__":

    #####################################
    # Shared temp folder
    #####################################
    test_data_path               = './dataset/processed/interpol-256/imagenet256-interp-dataset-T07052025.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T222343.hdf5' #'./dataset/processed/testset-256/imagenet256-testset-T151412.hdf5' # ./dataset/processed/testset-256/imagenet256-testset-T151633.hdf5'
    validation_data_path         = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'

    #####################################
    # Evaluation Parameters
    #####################################
    dataset_name        = "imagenet256-testset-T222343"
    group               = "test"  # "validation" or "test"
    batch_size          = 32
    samples_per_class   = 13
    num_pairs           = 12
    num_interpolations  = 20
    cfg_scale           = 3.0
    ccfg_scale          = 1.0
    sharpen_strength    = 0.1
    sharpen             = True

    #####################################
    # Model Paths
    #####################################
    
    # beta: 0.1
    #DiTSXL_Beta00x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA02x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-08/30859/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA05x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt'  
    DiTSXL_Beta05x05x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt'   ### (Baseline)
    
    # beta: 1.0
    #DITSXL_Beta05x05x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt'    
    DITSXL_Beta02x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt'   
    DITSXL_Beta05x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt'     

    # beta: 2.0
    DiTSXL_Beta02x10x_2b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_2.0b/BetaVAE-B-2/V0/2025-07-17/31522/checkpoints/last.ckpt'
    
    # beta: 5.0
    #DiTSXL_Beta05x05x_5b ='./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt'
    DiTSXL_Beta02x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt'    
    DiTSXL_Beta05x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt' 

    #####################################
    # Interpolation Class Pairs
    #####################################

    interpolation_dict = {
        "admiral_to_cabbage_butterfly": [321, 324],
        "cabbage_butterfly_to_monarch": [324, 323],
        "monarch_to_admiral_butterfly": [323, 321],
        # "lion_to_tiger": [291, 292],
        # "lion_to_leopard": [291, 288],
        # "leopard_to_tiger": [288, 292],
        # "leopard_to_lion": [288, 291],
        # "snow_leopard_to_tiger": [289, 292],
        # "snow_leopard_to_leopard": [289, 288],
        # "fox_to_tiger": [277, 292],
        # "fox_to_leopard": [277, 288],
        # "siamese_to_tiger": [284, 292],
        # "siamese_to_leopard": [284, 288],
        # "snow_leopard_to_siamese": [289, 284],
        # "red_panda_to_giant_panda": [387, 388],  # Kept, removed reversed
        "giant_panda_to_snow_leopard": [388, 289],
        "giant_panda_to_leopard": [388, 288],
        # "red_panda_to_lion": [387, 291],
        # "red_panda_to_tiger": [387, 292],
        # "husky_to_goldenretriever": [250, 207],
        # "goldenretriever_to_cockerspaniel": [207, 219],
        # "pembroke_corgi_to_cockerspaniel": [263, 219],
        "pembroke_corgi_to_husky": [263, 250],
        "fox_to_pembroke_corgi": [277, 263],
        # "cockatoo_to_lorikeet": [89, 90],
        # "macaw_to_cockatoo": [88, 89],
        "macaw_to_lorikeet": [88, 90],
        "horse_to_zebra": [339, 340],
        "horse_to_gazelle": [339, 353],
        "zebra_to_gazelle": [340, 353],
        "icebear_to_brownbear": [296, 294],
        "cockerspaniel_to_fox": [219, 277],
        "pembroke_corgi_to_fox": [263, 277],
        "grey_wolf_to_husky": [269, 250],
        "white_wolf_to_husky": [270, 250],
        "grey_wolf_to_white_wolf": [269, 270],
        "grey_wolf_to_fox": [269, 277],
        "brown_to_icebear": [294, 296],
        "brown_to_red_panda": [294, 387],
        # "giant_panda_to_brown_bear": [388, 294],
        # "icebear_to_red_panda": [296, 387],
    }

    # Updated selected_video_pairs with all unique interpolation names
    selected_video_pairs = list(interpolation_dict.keys())


    #####################################
    # Model Configurations
    #####################################
    model_configs = [
        # ---------------------------
        # beta: 0.1
        # ---------------------------
        # {
        #     "checkpoint": DiTSXL_Beta00x10x_01b,
        #     "source_timestep": 0.00,
        #     "target_timestep": 1.00,
        #     "beta": 0.1,
        #     "group_name": "Beta0.1"
        # },
        {
            "checkpoint": DITSXL_BETA05x10x_01b,
            "source_timestep": 0.50,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": DITSXL_BETA02x10x_01b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
        },
        {
            "checkpoint": DiTSXL_Beta05x05x_01b,
            "source_timestep": 0.50,
            "target_timestep": 0.50,
            "beta": 0.1,
            "group_name": "Beta0.1_Baseline"
        },
        # ---------------------------
        # beta: 1.0
        # ---------------------------
        # {
        #     "checkpoint": DITSXL_Beta05x05x_1b,
        #     "source_timestep": 0.50,
        #     "target_timestep": 0.50,
        #     "beta": 1.0,
        #     "group_name": "Beta1.0_Baseline"
        # },
        {
            "checkpoint": DITSXL_Beta02x10x_1b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 1.0,
            "group_name": "Beta1.0"
        },
        # {
        #     "checkpoint": DITSXL_Beta05x10x_1b,
        #     "source_timestep": 0.50,
        #     "target_timestep": 1.00,
        #     "beta": 1.0,
        #     "group_name": "Beta1.0"
        # },
        
        # ---------------------------
        # beta: 2.0
        # ---------------------------
        {
            "checkpoint": DiTSXL_Beta02x10x_2b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 2.0,
            "group_name": "Beta2.0"
        },
        # ---------------------------
        # beta: 5.0
        # ---------------------------
        # {
        #     "checkpoint": DiTSXL_Beta05x05x_5b,
        #     "source_timestep": 0.50,
        #     "target_timestep": 0.50,
        #     "beta": 5.0,
        #     "group_name": "Beta5.0_Baseline"
        # },
        {
            "checkpoint": DiTSXL_Beta02x10x_5b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 5.0,
            "group_name": "Beta5.0"
        },
        # {
        #     "checkpoint": DiTSXL_Beta05x10x_5b,
        #     "source_timestep": 0.50,
        #     "target_timestep": 1.00,
        #     "beta": 5.0,
        #     "group_name": "Beta5.0"
        # },
    ]


    #####################################
    # Device + Seed Setup
    #####################################
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()


    for config in model_configs:
        checkpoint = config["checkpoint"]
        source_timestep = config["source_timestep"]
        target_timestep = config["target_timestep"]
        beta = config["beta"]
        group_name = config["group_name"]
        baseline = (source_timestep == 0.50 and target_timestep == 0.50)
        model_name = f"{group_name}_VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"
        project_name = "CFM_Qualitative_Eval_Baseline" if baseline else "CFM_Qualitative_Eval"

        print(f"\n{'='*100}\nStarting evaluation for: {model_name}\n{'='*100}\n")
        # Prepare interpolation dictionary
        run_interpolation_and_evaluation(
            checkpoint=checkpoint,
            data_path=test_data_path if group == "test" else val_data_path,
            interpolation_dict=interpolation_dict,
            project_name=project_name,
            model_name=model_name,
            group=group,
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            beta=beta,
            samples_per_class=samples_per_class,
            num_interpolations=num_interpolations,
            selected_video_pairs=selected_video_pairs,
            num_pairs=num_pairs,
            cfg_scale=cfg_scale,
            ccfg_scale=ccfg_scale,
            batch_size=batch_size,
            sharpen_strength=sharpen_strength,
            sharpen=sharpen,
        )
        print(f"\nCompleted: {model_name}\n{'='*100}\n")
        torch.cuda.empty_cache()
        gc.collect()

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print("\n[INFO] Evaluation script completed successfully!")



# CUDA_VISIBLE_DEVICES=0 python ...










    
    

# if __name__ == "__main__":
    
#     #####################################
#     # Evaluation Parameters
#     #####################################
#     # Model checkpoints
#     source_timestep     = 0.50
#     target_timestep     = 1.00
#     beta                = 0.1     # Beta value for the VAE
#     dataset_name        = 'imagenet256-testset-T151412'
#     group               = "test"  # "validation" or "test"
#     baseline            = (source_timestep == 0.50 and target_timestep == 0.50)
#     batch_size          = 16
#     samples_per_class   = 14
#     num_pairs           = 12
#     num_interpolations  = 20
#     cfg_scale           = 4.0
#     ccfg_scale          = 1.0
    
#     #####################################
#     # Model Paths for SiT-XL-2
#     #####################################
    
#     # beta: 0.1
#     DiTSXL_Beta05x05x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt'   ### (Baseline)
#     DiTSXL_Beta00x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt'   ### Done
#     DITSXL_BETA02x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_0.1b/BetaVAE-B-2/2025-06-27/30400/checkpoints/last.ckpt'   ### Done
#     DITSXL_BETA05x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt'  
    
#     # beta: 1.0
#     DITSXL_Beta05x05x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt'    
#     DITSXL_Beta02x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt'   
#     DITSXL_Beta05x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt'          

#     # beta: 5.0
#     DiTSXL_Beta05x05x_5b ='./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt'
#     DiTSXL_Beta02x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt'    
#     DiTSXL_Beta05x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt'      
    
    
    
#     #####################################
#     # Dataset & Evaluation Parameters
#     #####################################
#     checkpoint                   = DITSXL_BETA05x10x_01b
#     test_data_path               = './dataset/processed/testset-256/imagenet256-testset-T151412.hdf5' # ./dataset/processed/testset-256/imagenet256-testset-T151633.hdf5'
#     validation_data_path         = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'
#     project_name                 = "CFM_Qualitative_Eval_Baseline" if baseline else "CFM_Qualitative_Eval"
#     model_name                   = f"Beta-VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"



#     #####################################
#     # Interpolation Class Pairs
#     #####################################
#     interpolation_dict = {
#         "trout_to_goldfish": [0, 1],
#         "admiral_to_cabbage_butterly": [321, 324],
#         "monarch_to_admiral_butterly": [323, 321],
#         "macaw_to_toucan": [88, 96],
#         "macaw_to_cockatoo": [88, 89],
#         "cockatoo_to_lorikeet": [89, 90],
#         "penguin_to_flamingo": [145, 130],
#         "lorikeet_to_peacock": [90, 84],
#         "flamingo_to_peacock": [130, 84],
#         "toucan_to_penguin": [96, 145],
#         "horse_to_zebra": [339, 340],
#         "camel_to_gazelle": [354, 353],
#         "gazelle_to_impalla": [353, 352],
#         "impalla_to_camel": [352, 354],
#         "siamese_to_persian_cat": [284, 283],
#         "snow_leopard_to_siamese_cat": [289, 284],
#         "red_panda_to_giant_panda": [387, 388],
#         "giant_panda_to_koala": [388, 105],
#         "koala_to_wombat": [105, 106],
#         "pembroke_corgi_to_cardigan_corgi": [263, 264],
#         "pembroke_corgi_to_cockerspaniel": [263, 219],
#         "fox_to_pembroke_corgi": [277, 263],
#         "husky_to_doberman": [250, 236],
#         "goldenretriever_to_chowchow": [207, 260],
#         "chowchow_to_cockerspaniel": [260, 219],
#         "cockerspaniel_to_fox": [219, 277],
#         "dalmatian_to_goldenretriever": [251, 207],
#         "japanese_spaniel_to_dalmatian": [152, 251],
#         "doberman_to_japanese_spaniel": [236, 152],
#         "fox_to_husky": [277, 250],
#         "lion_to_tiger": [291, 292],
#         "grey_wolf_to_white_wolf": [269, 270],
#         "white_wolf_to_husky": [270, 250],
#         "grey_wolf_to_husky": [269, 250],
#         "white_wolf_to_fox": [270, 277],
#         "snow_leopard_to_leopard": [289, 288],
#         "brown_to_icebear": [294, 296],
#         "icebear_to_brownbear": [296, 294],
#         "icebear_to_snow_leopard": [296, 289],
#         "icebear_to_white_wolf": [296, 270],
#         "gibbon_to_orangutan": [368, 365],
#         # Note: Add more class ID pairs
#     }
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(2025)

#     assert num_pairs <= samples_per_class, \
#         f"num_pairs ({num_pairs}) cannot be greater than samples_per_class ({samples_per_class})"


#     # Optional: Select specific pairs for evaluation
#     selected_video_pairs = [
#         "admiral_to_cabbage_butterly",
#         "monarch_to_admiral_butterly",
#         "macaw_to_toucan",
#         "macaw_to_cockatoo",
#         "lorikeet_to_peacock",
#         "toucan_to_penguin",
#         "grey_wolf_to_husky",
#         "cockatoo_to_lorikeet",
#         "flamingo_to_peacock",
#         "red_panda_to_giant_panda",
#         "goldenretriever_to_chowchow",
#         "chowchow_to_cockerspaniel",
#         "snow_leopard_to_leopard",
#         "icebear_to_brownbear",
#         "icebear_to_snow_leopard",
#         "icebear_to_white_wolf",
#         # Add more
#     ]


#     model_configs = {
        
#     }

#     for config in model_configs:
#         checkpoint = config["checkpoint"]
#         source_timestep = config["source_timestep"]
#         target_timestep = config["target_timestep"]
#         beta = config["beta"]
#         group_name = config["group_name"]

#         model_name = f"{group_name}_VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"
#         print(f"\n{'='*100}\nStarting evaluation for: {model_name}\n{'='*100}\n")
        
#         run_interpolation_and_evaluation(
#             checkpoint=checkpoint,
#             data_path=test_data_path if group == "test" else validation_data_path,
#             interpolation_dict=interpolation_dict,
#             project_name=project_name,
#             model_name=model_name,
#             device=device,
#             group=group,
#             source_timestep=source_timestep,
#             target_timestep=1.00,
#             beta=beta,
#             samples_per_class=samples_per_class,
#             num_interpolations=num_interpolations,
#             selected_video_pairs=selected_video_pairs,
#             num_pairs=num_pairs,
#             cfg_scale=cfg_scale,
#             ccfg_scale=ccfg_scale,
#             batch_size=batch_size,
#         )
        
#         print(f"\nCompleted: {model_name}\n{'='*100}\n")
#         torch.cuda.empty_cache()
#         gc.collect()
        
#     # Final cleanup
#     torch.cuda.empty_cache()
#     gc.collect()
#     print("\n[INFO] Evaluation script completed successfully!")


