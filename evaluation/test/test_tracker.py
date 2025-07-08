
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



############################################
# Image metrics tracker
############################################
class SmoothnessMetricsTracker(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS(net_type='alex').to(self.device)
        self.lpips.eval()
        self.reset()

    def reset(self):
        self.ppls = []
        self.istds = []

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
        
        print(f"[INFO] Processing batch of shape {batch.shape} for smoothness metrics.")


        for i in range(B):
            sequence = batch[i]  # (T, C, H, W)
            dists = []
            
            with torch.amp.autocast("cuda"):
                for t in range(T - 1):
                    d = self.lpips(sequence[t].unsqueeze(0), sequence[t + 1].unsqueeze(0)).item()
                    dists.append(d)

            if len(dists) == 0:
                print("[WARN] No valid LPIPS distances computed.")
                continue

            print(f"[DEBUG] Sequence {i} LPIPS distances: {dists}")

            self.ppls.append(float(torch.tensor(dists).mean()))
            self.istds.append(float(torch.tensor(dists).std()))

        print(f"[INFO] Processed {B} sequences for smoothness metrics.")
        

    @torch.no_grad()
    def aggregate(self):
        if not self.ppls:
            print("Warning: No data in tracker. Call update() before aggregate().")
            return {'ppl': float('nan'), 'istd': float('nan')}

        mean_ppl = torch.tensor(self.ppls).mean().item()
        mean_istd = torch.tensor(self.istds).mean().item()

        return {'ppl': mean_ppl, 'istd': mean_istd}




# class SmoothnessMetricsTracker(nn.Module):
#     """
#     Calculates smoothness metrics PPL and ISTD from a sequence of generated images.
#     Based on:
#     [0] PPL: "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
#     [1] Smooth Diffusion: "Crafting Smooth Latent Spaces in Diffusion Models" (Guo et al., 2024)
#     """
#     def __init__(self, device=None):
#         super().__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.lpips = LPIPS(net_type='alex').to(self.device)
#         self.lpips.eval()
#         self.reset()

#     def reset(self):
#         self.ppls = []
#         self.istds = []

#     @torch.no_grad()
#     def update(self, interpolated_imgs_batch):
#         assert interpolated_imgs_batch.dim() == 5, f"Expected shape (B, T, C, H, W), got {interpolated_imgs_batch.shape}"
#         B, T, C, H, W = interpolated_imgs_batch.shape
#         if T <= 1:
#             print("[WARN] Sequences must have at least two frames.")
#             return

#         # Normalize to [-1, 1] if needed
#         min_val, max_val = interpolated_imgs_batch.min(), interpolated_imgs_batch.max()
#         if min_val < -1.01 or max_val > 1.01:
#             min_per_seq = interpolated_imgs_batch.amin(dim=(2, 3, 4), keepdim=True)
#             max_per_seq = interpolated_imgs_batch.amax(dim=(2, 3, 4), keepdim=True)
#             denom = (max_per_seq - min_per_seq).clamp(min=1e-5)
#             interpolated_imgs_batch = 2 * (interpolated_imgs_batch - min_per_seq) / denom - 1

#         # Move to device
#         batch = interpolated_imgs_batch.to(self.device)

#         # Collect all pairs
#         all_pairs_1, all_pairs_2, seq_lengths = [], [], []

#         for i in range(B):
#             seq = batch[i]  # (T, C, H, W)
#             if seq.size(0) < 2:
#                 continue
#             all_pairs_1.append(seq[:-1].contiguous())
#             all_pairs_2.append(seq[1:].contiguous())
#             seq_lengths.append(seq.size(0) - 1)

#         if not all_pairs_1:
#             print("[WARN] No valid sequences found.")
#             return

#         # Concatenate all pairs
#         all_pairs_1 = torch.cat(all_pairs_1, dim=0)
#         all_pairs_2 = torch.cat(all_pairs_2, dim=0)

#         # Compute LPIPS distances in one vectorized pass
#         with torch.amp.autocast("cuda"):
#             dists_tensor = self.lpips(all_pairs_1, all_pairs_2)

#         dists_all = dists_tensor.flatten().cpu()

#         # Split back per sequence
#         idx = 0
#         for length in seq_lengths:
#             seq_dists = dists_all[idx:idx + length]
#             idx += length

#             # Important fix: avoid NaNs when only one pair
#             self.ppls.append(seq_dists.mean().item())
#             self.istds.append(seq_dists.std(unbiased=False).item())

#         print(f"[INFO] Processed {B} sequences with vectorized LPIPS.")

#     @torch.no_grad()
#     def aggregate(self):
#         if not self.ppls:
#             return {'ppl': float('nan'), 'istd': float('nan')}

#         return {
#             'ppl': torch.tensor(self.ppls).mean().item(),
#             'istd': torch.tensor(self.istds).mean().item()
#         }




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





class LatentSimilarityTracker(nn.Module):
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
            target, pred: tensors of shape [B, C, H, W] (decoded images) or [B, D] (latents).
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
            norm_target = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')
            norm_pred   = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')
            self.ssims.append(self.ssim(norm_pred, norm_target))
            self.psnrs.append(self.psnr(norm_pred, norm_target))
        else:
            # For latents: either skip PSNR or normalize entire vector
            norm_target = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')
            norm_pred   = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')
            self.psnrs.append(self.psnr(norm_pred, norm_target))

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



class ImageMetricsTracker(nn.Module):
    def __init__(self, num_crops: int = 1, crop_size: int = 256, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type='alex').to(self.device)
        self.lpips.eval()

        # Global FID
        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[ImageMetricsTracker] Evaluating using patch-wise FID")
            self.local_fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True
            ).to(self.device)

        self.num_crops = num_crops
        self.crop_size = crop_size

        self.reset()
        

    def update(self, target, pred, noise_target=None, noise_pred=None):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Convert to [0, 255] uint8
        real_ims_glb = un_normalize_ims(target) if target.max() <= 1 else target
        fake_ims_glb = un_normalize_ims(pred) if pred.max() <= 1 else pred

        real_ims_glb = real_ims_glb.clamp(0, 255).to(torch.uint8)
        fake_ims_glb = fake_ims_glb.clamp(0, 255).to(torch.uint8)

        ###################
        # Local (patch) FID
        ###################
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []

            real_imgs = real_ims_glb
            fake_imgs = fake_ims_glb

            for i in range(real_imgs.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                    real_imgs[0], output_size=(self.crop_size, self.crop_size)))

            for idx, (img_real, img_fake) in enumerate(zip(real_imgs, fake_imgs)):
                for i in range(self.num_crops):
                    anchor = anchors[idx * self.num_crops + i]
                    croped_real.append(FT.crop(img_real, *anchor))
                    croped_fake.append(FT.crop(img_fake, *anchor))

            real_ims_patches = torch.stack(croped_real)
            fake_ims_patches = torch.stack(croped_fake)

            with torch.amp.autocast("cuda"):
                self._check_fid_inputs(real_ims_patches, "real_ims_patches")
                self._check_fid_inputs(fake_ims_patches, "fake_ims_patches")
                self.local_fid.update(real_ims_patches, real=True)
                self.local_fid.update(fake_ims_patches, real=False)

        ###################
        # Global FID
        ###################
        with torch.amp.autocast("cuda"):
            self._check_fid_inputs(real_ims_glb, "real_ims_glb")
            self._check_fid_inputs(fake_ims_glb, "fake_ims_glb")
            self.global_fid.update(real_ims_glb, real=True)
            self.global_fid.update(fake_ims_glb, real=False)

        # Normalize pred and target for pixel metrics
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')

        with torch.amp.autocast("cuda"):
            self.ssims.append(self.ssim(pred_norm, target_norm))
            self.psnrs.append(self.psnr(pred_norm, target_norm))
            self.mses.append(torch.mean((pred_norm - target_norm) ** 2, dim=[1, 2, 3]))
            self.maes.append(torch.mean(torch.abs(pred_norm - target_norm), dim=[1, 2, 3]))
            self.lpips_scores.append(self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1))

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.maes = []
        self.lpips_scores = []
        self.global_fid.reset()
        if self.patch_fid:
            self.local_fid.reset()

    def aggregate(self):
        return dict(
            gfid=self.global_fid.compute().item(),
            lfid=self.local_fid.compute().item() if self.patch_fid else None,
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
