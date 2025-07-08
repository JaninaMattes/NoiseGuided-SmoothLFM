
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/mingukkang/elatentlpips

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
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)

from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

from ldm.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor



torch.set_float32_matmul_precision('high')



#########################################################
#                    Metric Tracker Classes             #
#########################################################

class SmoothnessMetricsTracker(nn.Module):
    """
    Calculates smoothness metrics PPL and ISTD from a sequence of generated images.
    This class is now decoupled from the image generation process.
    
    [0] Perceptual Path Length (PPL): 'Analyzing and Improving the Image Quality of StyleGAN'; (Tero Karras et al., 2020) – arXiv:1912.04958
    [1] Smooth Diffusion: 'Crafting Smooth Latent Spaces in Diffusion Models'; (Jiayi Guo et al., 2024) – arXiv:2312.04410
    
    Code adapted from: 
        - https://github.com/SHI-Labs/Smooth-Diffusion
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.lpips.eval()
        self.reset()

    def reset(self):
        self.latent_cppls = []
        self.latent_istds = []


    @torch.no_grad()
    def update(self, interpolated_imgs_batch):
        assert interpolated_imgs_batch.dim() == 5, f"Expected 5D tensor, got {interpolated_imgs_batch.dim()}D"
        B, T, C, H, W = interpolated_imgs_batch.shape
        assert T > 1, "Each sequence must contain at least 2 images."

        # Normalize only if needed (i.e., not already in [-1, 1])
        min_val, max_val = interpolated_imgs_batch.min(), interpolated_imgs_batch.max()
        if min_val < -1.01 or max_val > 1.01:
            print(f"[WARN] Input outside LPIPS range [{min_val:.3f}, {max_val:.3f}]. Normalizing.")
            
            # Rescale to [-1, 1] dynamically per sequence
            min_per_seq = interpolated_imgs_batch.amin(dim=(2, 3, 4), keepdim=True)
            max_per_seq = interpolated_imgs_batch.amax(dim=(2, 3, 4), keepdim=True)
            denom = (max_per_seq - min_per_seq).clamp(min=1e-5)
            interpolated_imgs_batch = 2 * (interpolated_imgs_batch - min_per_seq) / denom - 1

            
        batch = interpolated_imgs_batch.to(self.device)

        for i in range(B):
            sequence = batch[i]  # (T, C, H, W)
            dists = []
            for t in range(T - 1):
                d = self.lpips(sequence[t].unsqueeze(0), sequence[t + 1].unsqueeze(0)).item()
                dists.append(d)

            if len(dists) == 0:
                print("[WARN] No valid LPIPS distances computed.")
                continue

            self.latent_cppls.append(float(torch.tensor(dists).mean()))
            self.latent_istds.append(float(torch.tensor(dists).std()))

        print(f"[INFO] Processed {B} sequences for smoothness metrics.")


    @torch.no_grad()
    def aggregate(self):
        if not self.latent_cppls:
            print("Warning: No data in tracker. Call update() before aggregate().")
            return {'ppl': float('nan'), 'istd': float('nan')}

        mean_ppl = torch.tensor(self.latent_cppls).mean().item()
        mean_istd = torch.tensor(self.latent_istds).mean().item()

        return {'ppl': mean_ppl, 'istd': mean_istd}





class LatentSmoothnessTracker(nn.Module):
    """
    Computes smoothness metrics for latent sequences using cosine similarity and MSE (L2).
    Supports shapes (B, T, D, H, W) or (B, T, D).

    These metrics are latent-space adaptations of PPL and ISTD:
    - Original PPL (Perceptual Path Length) measures LPIPS distances between images generated from small latent interpolations [Karras et al., CVPR 2019].
    - Original ISTD (Interpolation Standard Deviation) quantifies pixel-level L2 variance in interpolated samples [Guo et al., CVPR 2024].
    
    
    As a variant we instead measure smoothness directly in latent space without decoding to pixel-space images.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        """Reset all stored pairwise distances."""
        self.all_mse_dists = []
        self.all_cos_dists = []

    @torch.no_grad()
    def update(self, latent_seq_batch):
        """
        Update metrics using a batch of latent sequences.

        Args:
            latent_seq_batch: (B, T, D) or (B, T, D, H, W)
        """
        assert latent_seq_batch.dim() in [3, 5], f"Expected 3D or 5D input, got {latent_seq_batch.dim()}D."
        B, T = latent_seq_batch.shape[:2]
        assert T > 1, "At least two timesteps required."

        batch = latent_seq_batch.to(self.device).float()
        if batch.dim() == 5:
            batch = batch.view(B, T, -1)  # flatten spatial dims

        for i in range(B):
            sequence = batch[i]  # shape (T, D_flat)

            for t in range(T - 1):
                seq1 = sequence[t]
                seq2 = sequence[t + 1]

                # Cosine distance
                sim = F.cosine_similarity(seq1, seq2, dim=0)
                cos_dist = max(0.0, 1.0 - sim.item())
                self.all_cos_dists.append(cos_dist)

                # MSE distance
                mse = F.mse_loss(seq1, seq2, reduction='mean')
                self.all_mse_dists.append(mse.item())

    @torch.no_grad()
    def aggregate(self):
        """Aggregate results and return global mean and std of distances."""
        if not self.all_mse_dists:
            return {
                'latent_mdpl': float('nan'),
                'latent_cdpl': float('nan'),
                'latent_mistd': float('nan'),
                'latent_cistd': float('nan')
            }

        all_mse_tensor = torch.tensor(self.all_mse_dists, device=self.device)
        all_cos_tensor = torch.tensor(self.all_cos_dists, device=self.device)

        return {
            'latent_mdpl': all_mse_tensor.mean().item(),
            'latent_cdpl': all_cos_tensor.mean().item(),
            'latent_mistd': all_mse_tensor.std().item(),
            'latent_cistd': all_cos_tensor.std().item()
        }




class ImageMetricsTracker(nn.Module):
    def __init__(self, num_crops: int=1, crop_size: int=256, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)                     # SSIM - requires [0, 1] range
        self.psnr = PSNR(data_range=1.0).to(self.device)                     # PSNR - requires [0, 1] range
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type='alex').to(self.device)  # expects pixel values in [-1, 1]
        self.lpips.eval()

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        # whether use the fid on crops during training
        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[ImageMetricTracker] Evaluating using patch-wise FID")
        self.num_crops = num_crops
        self.crop_size = crop_size

        # initialize
        self.reset()

    

    def __call__(self, target, pred, noise_target=None, noise_pred=None):
        """ Assumes target and pred in discretised range [0, 255] range 
            if in [0, 1] range, it will be converted to [0, 255]
        """   
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        print(f"Min: {target.min().item():.3f}, Max: {target.max().item():.3f} | Min: {pred.min().item():.3f}, Max: {pred.max().item():.3f}")
             
        # Data range [-1, 1] -> [0, 255] 
        real_ims = un_normalize_ims(target) if target.max() <= 1 else target
        fake_ims = un_normalize_ims(pred) if pred.max() <= 1 else pred
        
        real_ims = real_ims.clamp(0, 255).to(torch.uint8)
        fake_ims = fake_ims.clamp(0, 255).to(torch.uint8)
        
        ###################
        #  Pixel-space    #
        ################### 
        
        # FID  
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []
            for i in range(real_ims.shape[0]*self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                        real_ims[0], output_size=(self.crop_size, self.crop_size)))
                
            for idx, (img_real, img_fake) in enumerate(zip(real_ims, fake_ims)):
                for i in range(self.num_crops):
                    anchor = anchors[idx*self.num_crops + i]

                    croped_real.append(FT.crop(img_real, *anchor))
                    croped_fake.append(FT.crop(img_fake, *anchor))

            real_ims = torch.stack(croped_real)
            fake_ims = torch.stack(croped_fake)
            
        self._check_fid_inputs(real_ims, "real_ims")
        self._check_fid_inputs(fake_ims, "fake_ims")
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        # Normalize: [-1, 1] -> [0, 1]   
        pred = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')                                
        target = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')

        # SSIM, LPIPS, PSNR, MAE and MSE 
        self.ssims.append(self.ssim(pred, target))                              # SSIM expects [0, 1] range, or None for self-determined range
        self.psnrs.append(self.psnr(pred, target))                              # PSNR expects [0, 1] range, or None for self-determined range
        self.mses.append(torch.mean((pred - target) ** 2, dim=[1, 2, 3]))       # MSE  expects [0, 1] range
        self.maes.append(torch.mean(torch.abs(pred - target), dim=[1, 2, 3]))   # MAE  expects [0, 1] range

        # LPIPS range [0, 1] -> [-1, 1]
        self.lpips_scores.append(self.lpips(pred * 2 - 1, target * 2 - 1))      # LPIPS expects [-1, 1] range
        

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.maes = []
        self.lpips_scores = []
        self.fid.reset() 



    def aggregate(self):
        fid_result = self.fid.compute() # returns dict
        return dict(
            fid=fid_result.item(),
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item()
        )

    def _check_fid_inputs(self, tensor, name):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a tensor."
        assert tensor.dtype == torch.uint8, f"{name} must be uint8 but got {tensor.dtype}."
        assert tensor.dim() == 4 and tensor.size(1) == 3, f"{name} must have shape (N, 3, H, W), got {tensor.shape}."
        assert tensor.min() >= 0 and tensor.max() <= 255, f"{name} must be in [0, 255], got range [{tensor.min()}, {tensor.max()}]."




class LatentSimilarityTracker(nn.Module):
    """ Input shape: B, 4, 32, 32
    Computes pixel-space similarity metrics (SSIM, PSNR, MSE, MAE) and cosine similarity
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pixel-space similarity metrics
        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)

        self.reset()

    def reset(self):
        self.cosine_sims = []
        self.psnrs = []
        self.ssims = []
        self.mses = []
        self.maes = []

    def update(self, target, pred):
        """
        Args:
            target, pred: tensors of shape [B, C, H, W] (latents with spatial dimensions, B, 4, 32, 32).
                          Assumed to be in [0, 1] range for SSIM/PSNR.
        """
        assert target.shape == pred.shape, f"Shape mismatch: {target.shape} vs {pred.shape}"
        target = target.to(self.device).float()
        pred = pred.to(self.device).float()
            
        # Cosine (on raw latents)
        flat_target = target.view(target.size(0), -1)
        flat_pred = pred.view(pred.size(0), -1)
        self.cosine_sims.append(F.cosine_similarity(flat_target, flat_pred, dim=1))

        # Optional: normalize for PSNR / SSIM
        if target.size(1) == 3:
            # Normalize: [-1, 1] -> [0, 1]   
            pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')                                
            target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')
            self.ssims.append(self.ssim(pred_norm, target_norm))
            self.psnrs.append(self.psnr(pred_norm, target_norm))
        else:
            # For latents: either skip PSNR or normalize entire vector
            pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')
            target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')
            self.psnrs.append(self.psnr(pred_norm, target_norm))

        # MSE / MAE on raw
        dim = 1 if target.ndim == 2 else [1, 2, 3]
        self.mses.append(torch.mean((pred - target) ** 2, dim=dim))
        self.maes.append(torch.mean(torch.abs(pred - target), dim=dim))

        
    
    def aggregate(self):
        return dict(
            cosine=torch.cat(self.cosine_sims).mean().item(),
            ssim=torch.stack(self.ssims).mean().item() if self.ssims else float("nan"),
            psnr=torch.stack(self.psnrs).mean().item() if self.psnrs else float("nan"),
            mse=torch.cat(self.mses).mean().item(),
            mae=torch.cat(self.maes).mean().item()
        )


    

#########################################################
#                    Interpolation                      #
#########################################################

@torch.no_grad()
def interpolate_vectors(z1, z2, alpha_vals, mode="linear", dot_threshold=0.9995):
    """
    Interpolates between z1 and z2 using either 'linear' or 'slerp' mode.
    Returns a tensor of shape (B, D) where B = len(alpha_vals).
    """
    
    if alpha_vals.numel() == 0:
        raise ValueError(f"[interpolate_vectors] alpha_vals is empty. Ensure num_interpolations > 0.")

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





@torch.no_grad()
def interpolate_latents_with_smoothness_eval(
    cls1_latents,
    cls2_latents,
    cls1_images,
    cls2_images,
    cls1_label,
    cls2_label,
    beta_vae_module,
    smoothness_tracker,
    num_pairs=10,
    num_interpolations=16,
    cfg_scale=1.0,
    ccfg_scale=0.0,
    sample_kwargs=None,
    use_labels=False,
    num_classes=1000,
    device=None,
    interp_type='linear',
    upscale_to=128,
    plot_samples=False,
    save_dir=None,
    title=None,
    gt_border=0,
    line_width=5, 
):
    device = device or beta_vae_module.device
    alpha_lin_space = torch.linspace(0, 1, num_interpolations).to(device)

    all_sequences = []
    generated_rows = []
    smoothness_tracker.reset()

    for i in tqdm(range(num_pairs), desc="Interpolating Latents", leave=False):
        x1 = cls1_latents[i].unsqueeze(0).to(device)
        x2 = cls2_latents[i].unsqueeze(0).to(device)

        context_z1 = beta_vae_module.model.encode(x1)['latent_dist'].sample().squeeze(0)
        context_z2 = beta_vae_module.model.encode(x2)['latent_dist'].sample().squeeze(0)

        interp_context = interpolate_vectors(context_z1.to(device), context_z2.to(device), alpha_vals=alpha_lin_space, mode=interp_type)

        B = interp_context.shape[0]
        labels = torch.cat([
            torch.full((B // 2,), cls1_label, dtype=torch.long),
            torch.full((B - B // 2,), cls2_label, dtype=torch.long),
        ]).to(device) if use_labels else None

        decoded_interpolation = beta_vae_module.model.decode(interp_context)["sample"]
        all_sequences.append(decoded_interpolation.unsqueeze(0))

        if plot_samples:
            decoded_latents = beta_vae_module.decode_second_stage(decoded_interpolation.to(device), label=labels)
            generated_interpolation = beta_vae_module.decode_first_stage(decoded_latents.to(device))

            # Denorm is fine for visualization, but not for metrics
            row_images = denorm_tensor(generated_interpolation * 0.5 + 0.5).detach().cpu()
            row_images = torch.stack([TF.resize(img, [upscale_to, upscale_to]) for img in row_images])
    
            real_start = denorm_tensor(cls1_images[i].unsqueeze(0))[0].detach().cpu()
            real_start = TF.resize(real_start, [upscale_to, upscale_to])
            real_end = denorm_tensor(cls2_images[i].unsqueeze(0))[0].detach().cpu()
            real_end = TF.resize(real_end, [upscale_to, upscale_to])

            real_start_padded = F.pad(real_start, pad=[0, gt_border, 0, 0], mode='constant', value=0)
            real_end_padded = F.pad(real_end, pad=[gt_border, 0, 0, 0], mode='constant', value=0)

            pad_left = gt_border // 2
            pad_right = gt_border - pad_left
            row_images_padded = torch.stack([
                F.pad(img, pad=[pad_left, pad_right, 0, 0], mode='constant', value=0)
                for img in row_images
            ])

            full_row = torch.cat([real_start_padded.unsqueeze(0), row_images_padded, real_end_padded.unsqueeze(0)], dim=0)
            generated_rows.append(full_row)
            

    if plot_samples and save_dir is not None and generated_rows:
        all_rows = torch.cat(generated_rows, dim=0)  # shape: (num_pairs * (T+2), C, H, W)
        nrow = num_interpolations + 2
        grid = make_grid(all_rows, nrow=nrow, padding=0)
        grid_np = grid.permute(1, 2, 0).numpy()

        rcParams.update({'font.size': 14, 'font.family': 'DejaVu Sans'})
        fig, ax = plt.subplots(figsize=(grid.shape[2] / 50, grid.shape[1] / 50))
        ax.imshow(grid_np)
        ax.axis('off')

        # === FIXED: Only draw vertical lines after first and before last ===
        img_width = grid_np.shape[1] / nrow
        ax.axvline(x=img_width, color='black', linewidth=line_width)            # After Real A
        ax.axvline(x=(nrow - 1) * img_width, color='black', linewidth=line_width)  # Before Real B

        xtick_positions = [(i + 0.5) * img_width for i in range(nrow)]
        xtick_labels = ['Real A'] + [f'{alpha:.2f}' for alpha in alpha_lin_space.tolist()] + ['Real B']
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45)

        ytick_positions = [(i + 0.5) * (grid_np.shape[0] / len(generated_rows)) for i in range(len(generated_rows))]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([f'Pair {i}' for i in range(len(generated_rows))], fontsize=10)

        title = title or f"Latent Interpolation Grid ({interp_type})"
        ax.set_title(title, fontsize=14)

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.tight_layout()
        save_dir = os.path.join(save_dir, f"{interp_type}_cls1_{cls1_label}_cls2_{cls2_label}.png")
        plt.savefig(save_dir, bbox_inches='tight', dpi=500)
        print(f"[INFO] Saved interpolation grid to: {save_dir}")
        plt.show()
        plt.close(fig)

        del grid, grid_np, fig, ax, all_rows
        del generated_rows, row_images, real_start, real_end, real_start_padded, real_end_padded, row_images_padded
        torch.cuda.empty_cache()
        gc.collect()

    # === Final: Smoothenss Evaluation ===
    all_sequences = torch.cat(all_sequences, dim=0)  # shape: (num_pairs, T, C, H, W)
    smoothness_tracker.update(all_sequences)
    
    return all_sequences.cpu()





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
    images,
    tracker,
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
    images = images.to(device)
    xt_latents = xt_latent.to(device)
    
    if labels is not None:
        labels = labels.to(device).squeeze()

    with torch.no_grad():
        # Encode-Decode with ß-VAE
        latents = beta_vae_module.model.encode(xt_latents)['latent_dist'].sample()
        fake_latents = beta_vae_module.model.decode(latents)['sample']

        # Track latents
        tracker.update(xt_latents.to(device), fake_latents.to(device))

    # Plotting (optional)
    if plot_samples:
        
        # Decode to RGB space
        latent = beta_vae_module.decode_second_stage(fake_latents, label=labels)
        fake_images = beta_vae_module.decode_first_stage(latent)
        real_images = images  # Already unnormalized

        real_images_ = denorm_tensor(real_images).detach().cpu()
        fake_images_ = denorm_tensor(fake_images).detach().cpu()
        real_images_resized = TF.resize(real_images_, [resize_to, resize_to])
        fake_images_resized = TF.resize(fake_images_, [resize_to, resize_to])

        interleaved = []
        for real, fake in zip(real_images_resized, fake_images_resized):
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

        del grid, fake_images_, real_images_, real_images_resized, fake_images_resized
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    return fake_latents.detach().cpu(), xt_latents.detach().cpu()




#########################################################
#                    Collect Samples                    #
#########################################################
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
            raise ValueError(f"Not enough samples collected for class {label}: got {len(latents_list)}")

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
#                         RUN                           #
#########################################################
def run_quant_eval(
    checkpoint,
    data_path,
    interpolation_dict,
    project_name,
    model_name,
    group="validation",
    source_timestep=0.50,
    target_timestep=1.00,
    beta=0.1,
    samples_per_class=10,
    num_pairs=5,
    num_interpolations=16,
    cfg_scales=[1.0, 2.0, 3.0, 4.0],
    ccfg_scales=[1.0, 1.0, 1.0, 1.0],
    batch_size=32,
    dataset_name="imagenet256-testset-T123052.hdf5",
    results_root="results",
    max_samples=50000,
    num_steps=50,
    num_classes=1000,
    plot_every_n_batches=1000,
    device=None,
):
    assert len(cfg_scales) == len(ccfg_scales), "cfg_scales and ccfg_scales must have the same length."
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model
    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(checkpoint, map_location='cpu')
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

    num_params = sum(p.numel() for p in beta_vae_module.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(f"[INFO] Running evaluation for group: {group}, model: {model_name}, dataset: {dataset_name}")
    
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
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {base_results_dir}")  
    
    # CSV setup
    csv_path = base_results_dir / f"{model_name}_metrics.csv"
    with open(csv_path, "w") as csv_file:
        csv_file.write("evaluation_type,model_name,dataset,CFG,CCFG,SSIM,PSNR,MSE,COS,MSE-DPL,MSE-ISTD,Cos-DPL,Cos-ISTD,interpolation_pair\n")
        
        
        
        ##############################
        # PART 1: Image Quality
        ##############################
        print("\n--- Part 1: Image Quality Evaluation ---")
        tracker = LatentSimilarityTracker(device=device)

        for cfg_scale, ccfg_scale in zip(cfg_scales, ccfg_scales):
            print(f"[INFO] Evaluating CFG={cfg_scale}, CCFG={ccfg_scale}")
            tracker.reset()

            dataloader = get_dataloader_by_group(data, group)
            count = 0

            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                if batch["image"].size(0) < batch_size:
                    print(f"[WARN] Skipping batch {batch_idx} with size {batch['image'].size(0)} < {batch_size}")
                    continue

                if count >= max_samples:
                    print(f"[INFO] Reached max sample limit: {max_samples}")
                    break

                images = batch["image"]
                xt = batch[f"latents_{source_timestep:.2f}"]
                labels = batch["label"]

                plot_now = (batch_idx % plot_every_n_batches == 0)
                plot_path = str(base_results_dir / f"batch_{batch_idx}_cfg{cfg_scale}_ccfg{ccfg_scale}.png")
                title_str = f"Generated Samples (CFG={cfg_scale}, CCFG={ccfg_scale})"

                fake_imgs, real_imgs = generate_samples(
                    beta_vae_module,
                    tracker=tracker,
                    images=images,
                    xt_latent=xt,
                    labels=labels,
                    cfg_scale=cfg_scale,
                    ccfg_scale=ccfg_scale,
                    num_steps=num_steps,
                    num_classes=num_classes,
                    plot_samples=plot_now,
                    save_path=plot_path if plot_now else None,
                    title=title_str,
                    device=device,
                )

                count += real_imgs.size(0)

                del real_imgs, fake_imgs, images, xt, labels
                torch.cuda.empty_cache()
                gc.collect()

            metrics = tracker.aggregate()
            print(f"[INFO] CFG={cfg_scale}, CCFG={ccfg_scale} → MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.6f}, COS: {metrics['cosine']:.6f}")

            csv_file.write(
                f"image_quality,{model_name},{dataset_name},{cfg_scale},{ccfg_scale},"
                f"{metrics['ssim']:.6f},{metrics['psnr']:.6f},{metrics['mse']:.6f},{metrics['cosine']:.6f},"
                f"nan,nan,nan,nan,nan\n"
            )
            csv_file.flush()



        ##############################
        # PART 2: Smoothness Metrics
        ##############################
        print("\n--- Part 2: Interpolation Smoothness Evaluation ---")
        smoothness_tracker = LatentSmoothnessTracker(device=device)

        all_classes = {cls for pair in interpolation_dict.values() for cls in pair}
        latents, labels, images = collect_samples(
            data=data,
            class_labels=list(all_classes),
            source_timestep=source_timestep,
            samples_per_class=max(samples_per_class, num_pairs),
            group_name=group,
        )

        global_mdpl, global_cdpl = 0.0, 0.0  # MSE-based, Cosine-based path lengths
        global_mistd, global_cistd = 0.0, 0.0  # MSE-based, Cosine-based std deviations

        count_interp = 0
        for interp_name, (cls_a, cls_b) in tqdm(interpolation_dict.items(), desc="Processing Interpolations"):
            smoothness_tracker.reset()
            
            cls1_latents = latents[labels == cls_a][:num_pairs]
            cls2_latents = latents[labels == cls_b][:num_pairs]
            cls1_images = images[labels == cls_a][:num_pairs]
            cls2_images = images[labels == cls_b][:num_pairs]

            print(f"[INFO] Interpolating: {interp_name} ({cls_a} → {cls_b})")
            plot_now = (count_interp % 5 == 0)

            interpolate_latents_with_smoothness_eval(
                cls1_latents=cls1_latents,
                cls2_latents=cls2_latents,
                cls1_images=cls1_images,
                cls2_images=cls2_images,
                cls1_label=cls_a,
                cls2_label=cls_b,
                beta_vae_module=beta_vae_module,
                smoothness_tracker=smoothness_tracker,
                num_pairs=num_pairs,
                num_interpolations=num_interpolations,
                cfg_scale=3.0,
                ccfg_scale=1.0,
                sample_kwargs={"num_steps": num_steps},
                use_labels=False,
                num_classes=num_classes,
                device=device,
                interp_type='linear',
                plot_samples=plot_now,
                save_dir=base_results_dir,
                title=f"Synthesized Interpolation {interp_name} ({cls_a} → {cls_b})",
            )
            count_interp += 1
            
            out = smoothness_tracker.aggregate()
            mean_mdpl, mean_cdpl, mean_mistd, mean_cistd = out['latent_mdpl'], out['latent_cdpl'], out['latent_mistd'], out['latent_cistd']
            print(f"[INFO] {interp_name} → MSE-DPL: {mean_mdpl:.6f}, MSE-ISTD: {mean_mistd:.6f}, Cos-DPL: {mean_cdpl:.6f}, Cos-ISTD: {mean_cistd:.6f}")

            global_mdpl += mean_mdpl
            global_cdpl += mean_cdpl
            global_mistd += mean_mistd
            global_cistd += mean_cistd
            
            csv_file.write(
                f"smoothness,{model_name},{dataset_name},nan,nan,nan,nan,nan,nan,"
                f"{mean_mdpl:.6f},{mean_mistd:.6f},{mean_cdpl:.6f},{mean_cistd:.6f},{interp_name}\n"
            )   
            csv_file.flush()
            
            torch.cuda.empty_cache()
            gc.collect()

        avg_mdpl = global_mdpl / len(interpolation_dict)
        avg_cdpl = global_cdpl / len(interpolation_dict)
        avg_mistd = global_mistd / len(interpolation_dict)
        avg_cistd = global_cistd / len(interpolation_dict)

        csv_file.write(
            f"smoothness_average,{model_name},{dataset_name},nan,nan,nan,nan,nan,nan,"
            f"{avg_mdpl:.6f},{avg_mistd:.6f},{avg_cdpl:.6f},{avg_cistd:.6f},ALL_PAIRS\n"
        )
        csv_file.flush()

        print(f"\nInterpolation complete. Avg Cos-DPL: {avg_cdpl:.6f}, Avg Cos-ISTD: {avg_cistd:.6f}, Avg MSE-DPL: {avg_mdpl:.6f}, Avg MSE-ISTD: {avg_mistd:.6f}")




#########################################################
#                         END RUN                       #
#########################################################





if __name__ == "__main__":

    #####################################
    # Evaluation Setup
    #####################################
    source_timestep = 0.20
    target_timestep = 1.00
    beta            = 3.0  # Beta value for the VAE
    dataset_name    = 'imagenet256-dataset-T000006'
    group           = "validation"  # or "test"
    baseline        = (source_timestep == 0.50 and target_timestep == 0.50)



    #####################################
    # Model Setup
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
    Beta02x10x_05b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt'

    # beta: 1.0
    Beta05x05x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
    Beta05x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'                                                                                                                                   # Open
    Beta02x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'                                          # Open

    # beta: 2.0
    Beta02x10x_2b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-2.0b/V0/2025-07-02/101646/checkpoints/last.ckpt'                     # Open
    
    
    # beta: 3.0
    Beta02x10x_3b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt'                     # Open

    # beta: 5.0
    Beta05x05x_5b ='./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
    Beta05x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'                                                                                                                                  # Open
    Beta02x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'                   # Open


    #####################################
    # Dataset & Evaluation Parameters
    #####################################
    checkpoint      = Beta02x10x_3b
    data_path       = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'
    project_name    = "BetaVAE_Quantitative_Eval_Baseline" if baseline else "BetaVAE_Quantitative_Eval"
    model_name      = f"Beta-VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"
    
    

    #####################################
    # Interpolation Class Pairs
    #####################################
    interpolation_dict = {
        "admiral_to_cabbage_butterly": [321, 324],
        "monarch_to_admiral_butterfly": [323, 321],
        "siamese_to_persian_cat": [284, 283],
        "red_panda_to_giant_panda": [387, 388],
        "pembroke_corgi_to_cardigan_corgi": [263, 264],
        "husky_to_doberman": [250, 236],
        "grey_to_white_wolf": [269, 270],
        "horse_to_zebra": [339, 340],
        "camel_to_impala": [339, 353],
        "lion_to_tiger": [291, 292],
        "snow_leopard_to_leopard": [289, 288],
        "brown_to_icebear": [294, 296],
        "gibbon_to_orangutan": [368, 365],  
        "lorikeet_to_peacock": [90, 84],
        "macaw_to_toucan": [88, 96],
        "penguin_to_cockatoo": [145, 89],
        # Note: Add more class ID pairs
    }

    #####################################
    # Device + Seed Setup
    #####################################
    torch.manual_seed(2025)


    samples_per_class           = 10        # Number of samples per class for interpolation
    num_pairs                   = 10        # Number of pairs to interpolate between classes
    num_interpolations          = 24        # Number of interpolation steps between each pair
    max_samples                 = 1000     # Maximum number of samples to process in total (50k for FID, etc.)
    batch_size                  = 32        # Batch size for evaluation

    # CFG scaled don't matter in ß-VAE evaluation, but we keep them for consistency
    # INFO: Increasing CFG has no effect on the ß-VAE, as it is not a diffusion model.
    cfg_scales                  = [1.0]  # [1.0, 2.0, 3.0, 4.0, 8.0, 10.0]  # [1.0] Context-Conditional CFG scales
    ccfg_scales                 = [1.0]  # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # [1.0] Class-conditional CFG scales


    #####################################
    # Run Evaluation
    #####################################
    run_quant_eval(
        checkpoint          = checkpoint,
        data_path           = data_path,
        interpolation_dict  = interpolation_dict,
        project_name        = project_name,
        model_name          = model_name,
        group               = group,
        source_timestep     = source_timestep,
        target_timestep     = target_timestep,
        beta                = beta,
        samples_per_class   = samples_per_class,
        num_pairs           = num_pairs,
        num_interpolations  = num_interpolations,
        cfg_scales          = cfg_scales,
        ccfg_scales        = ccfg_scales,
        batch_size         = batch_size,
        dataset_name       = dataset_name,
        results_root       = "results",
        max_samples        = max_samples, # 50 k for FID (50000 samples)
        num_steps          = 50,
        num_classes        = 1000
    )



# CUDA_VISIBLE_DEVICES=2 python '/export/home/ra93jiz/dev/Img-IDM/ldm/evaluation/beta_vae_quantitiative_eval.py'