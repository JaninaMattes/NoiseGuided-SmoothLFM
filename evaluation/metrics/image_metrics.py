
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

import numpy as np


from scipy import linalg
from matplotlib import pyplot as plt
from matplotlib import rcParams

# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from pytorch_fid.inception import InceptionV3


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
#   FID class using InceptionV3
############################################
class PrecisionRecallFID(nn.Module):
    def __init__(self, k=3, device=None):
        super().__init__()
        self.k = k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.device).eval()
        self.real_feats = []
        self.fake_feats = []

    @torch.no_grad()
    def update(self, images, real=True):
        x = images.to(self.device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x + 1) / 2 if x.min() < 0 else x  # Convert [-1, 1] to [0, 1]
        feats = self.inception(x)[0].squeeze(-1).squeeze(-1)  # Shape (N, 2048)
        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.no_grad()
    def compute_pFID_rFID(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)

        dists_real = self._pairwise(real_feats, real_feats)
        radii_real = dists_real.topk(self.k + 1, largest=False).values[:, -1]

        dists_fake = self._pairwise(fake_feats, fake_feats)
        radii_fake = dists_fake.topk(self.k + 1, largest=False).values[:, -1]

        dists_cross = self._pairwise(fake_feats, real_feats)

        precision_mask = (dists_cross <= radii_real.unsqueeze(0)).any(dim=1)
        recall_mask = (dists_cross.t() <= radii_fake.unsqueeze(0)).any(dim=1)

        fake_in_real = fake_feats[precision_mask]
        real_in_fake = real_feats[recall_mask]

        # pFID: fake_in_real vs real_feats
        pFID = self._compute_fid(real_feats, fake_in_real)

        # rFID: real_in_fake vs fake_feats
        rFID = self._compute_fid(fake_feats, real_in_fake)

        return pFID, rFID

    def _pairwise(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()



    def _compute_fid(self, feats1, feats2, eps=1e-6):
        mu1 = feats1.mean(dim=0)
        mu2 = feats2.mean(dim=0)

        sigma1 = self._cov(feats1)
        sigma2 = self._cov(feats2)

        diff = mu1 - mu2

        cov_prod = sigma1.cpu().numpy() @ sigma2.cpu().numpy()
        covmean = linalg.sqrtm(cov_prod)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1.cpu().numpy() + offset) @ (sigma2.cpu().numpy() + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        covmean = torch.from_numpy(covmean).to(sigma1.device)

        fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    def _cov(self, feats):
        feats_np = feats.cpu().numpy()
        cov = np.cov(feats_np, rowvar=False)
        return torch.from_numpy(cov).to(feats.device)





############################################
# Precision & Recall class using InceptionV3
############################################
class PrecisionRecall(nn.Module):
    """
    Precision & Recall metrics as defined in Kynkäänniemi et al.
        + Computes k-nearest neighbor distances.
        + Estimates precision (fraction of fake images inside real manifold).
        + Estimates recall (fraction of real images inside fake manifold).
    
        https://arxiv.org/abs/1904.06991
        https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
    """
    def __init__(self, k=3, device=None):
        super().__init__()
        self.k = k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_feats = []
        self.fake_feats = []

    def reset(self):
        self.real_feats = []
        self.fake_feats = []

    @torch.no_grad()
    def update(self, images, real=True):
        feats = images.to(self.device)

        # Ensure images are float before flattening
        if feats.dtype == torch.uint8:
            feats = feats.float() / 255.0

        feats = feats.view(feats.size(0), -1)  # flatten
        assert feats.dtype in (torch.float32, torch.float64), f"Expected float type, got {feats.dtype}"

        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.no_grad()
    def compute(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)

        # Compute pairwise distances between real samples (for recall manifold)
        dists_real = self._pairwise_distances(real_feats, real_feats)
        radii_real = dists_real.topk(self.k + 1, largest=False).values[:, -1]

        # Compute pairwise distances between fake samples (for precision manifold)
        dists_fake = self._pairwise_distances(fake_feats, fake_feats)
        radii_fake = dists_fake.topk(self.k + 1, largest=False).values[:, -1]

        # Cross distances: fake-to-real
        dists_cross = self._pairwise_distances(fake_feats, real_feats)
        precision_mask = (dists_cross <= radii_real.unsqueeze(0)).any(dim=1)
        precision = precision_mask.float().mean().item()

        # Cross distances: real-to-fake
        dists_cross_T = dists_cross.t()
        recall_mask = (dists_cross_T <= radii_fake.unsqueeze(0)).any(dim=1)
        recall = recall_mask.float().mean().item()

        return precision, recall

    def _pairwise_distances(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()

    

############################################
# Image metrics tracker
############################################
class ImageMetricsTracker(nn.Module):
    """
    With additional Precision-Recall:
        + Computes k-nearest neighbor distances.
        + Estimates precision (fraction of fake images inside real manifold).
        + Estimates recall (fraction of real images inside fake manifold).
    """
    def __init__(self, num_crops=4, crop_size=128, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.lpips.eval()

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        self.prec_recall = PrecisionRecall(k=3, device=self.device).to(self.device)
        self.prec_recall_fid = PrecisionRecallFID(k=3, device=self.device).to(self.device)

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

    def update(self, target, pred):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Convert to [0, 255] uint8 for FID and PR metrics
        real_ims_glb = denorm_metrics_tensor(target, target_range=(0, 255), dtype='int')
        fake_ims_glb = denorm_metrics_tensor(pred, target_range=(0, 255), dtype='int')
          
        def to_float01(x):
            if x.min() >= 0 and x.max() <= 1:
                return x
            elif x.min() >= -1 and x.max() <= 1:
                return (x + 1) / 2
            elif x.dtype == torch.uint8:
                return x.float() / 255
            else:
                raise ValueError(f"Unsupported input range: min {x.min().item()}, max {x.max().item()}")
        
        # Local (patch) FID
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []

            for i in range(real_ims_glb.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                    real_ims_glb[0], output_size=(self.crop_size, self.crop_size)))

            for idx, (img_real, img_fake) in enumerate(zip(real_ims_glb, fake_ims_glb)):
                for i in range(self.num_crops):
                    anchor = anchors[idx * self.num_crops + i]
                    croped_real.append(FT.crop(img_real, *anchor))
                    croped_fake.append(FT.crop(img_fake, *anchor))

            real_ims_patches = torch.stack(croped_real)
            fake_ims_patches = torch.stack(croped_fake)
            self.local_fid.update(real_ims_patches, real=True)
            self.local_fid.update(fake_ims_patches, real=False)

        self.global_fid.update(real_ims_glb, real=True)
        self.global_fid.update(fake_ims_glb, real=False)

        self.prec_recall.update(real_ims_glb, real=True)
        self.prec_recall.update(fake_ims_glb, real=False)

        self.prec_recall_fid.update(real_ims_glb, real=True)
        self.prec_recall_fid.update(fake_ims_glb, real=False)


        # Normalize pred and target for pixel metrics [0, 1]
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float')
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float')

        # Normalize for pixel-level metrics
        pred_norm = to_float01(pred)
        target_norm = to_float01(target)

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
        self.prec_recall.reset()
        if self.patch_fid:
            self.local_fid.reset()

    def aggregate(self):
        precision_val, recall_val = self.prec_recall.compute()
        pFID_val, rFID_val = self.prec_recall_fid.compute_pFID_rFID()
        
        # Clamping values to avoid negative metrics
        precision_val = max(precision_val, 0.0)
        recall_val = max(recall_val, 0.0)
        pFID_val = max(pFID_val, 0.0)
        rFID_val = max(rFID_val, 0.0)
        
        # Compute global FID
        gfid = self.global_fid.compute().item()
        gfid = max(gfid, 0.0)

        if self.patch_fid:
            lfid = self.local_fid.compute().item()
            lfid = max(lfid, 0.0)
        else:
            lfid = None


        return dict(
            gfid=gfid,
            lfid=lfid,
            precision=precision_val,
            recall=recall_val,
            pFID=pFID_val,
            rFID=rFID_val,
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item()
        )


    def _check_fid_inputs(self, tensor, name):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a tensor."
        assert tensor.dtype == torch.uint8, f"{name} must be uint8 but got {tensor.dtype}."
        assert tensor.dim() == 4 and tensor.size(1) == 3, f"{name} shape must be (N, 3, H, W), got {tensor.shape}."
        assert tensor.min() >= 0 and tensor.max() <= 255, f"{name} must be in [0, 255], got [{tensor.min()}, {tensor.max()}]."








############################################
# Precision & Recall class using InceptionV3
############################################
class PrecisionRecall(nn.Module):
    """
    Precision & Recall metrics as defined in Kynkäänniemi et al.
        + Computes k-nearest neighbor distances.
        + Estimates precision (fraction of fake images inside real manifold).
        + Estimates recall (fraction of real images inside fake manifold).
    
        https://arxiv.org/abs/1904.06991
        https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
    """
    def __init__(self, k=3, device=None):
        super().__init__()
        self.k = k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_feats = []
        self.fake_feats = []

    def reset(self):
        self.real_feats = []
        self.fake_feats = []

    @torch.no_grad()
    def update(self, images, real=True):
        feats = images.to(self.device)

        # Ensure images are float before flattening
        if feats.dtype == torch.uint8:
            feats = feats.float() / 255.0

        feats = feats.view(feats.size(0), -1)  # flatten
        assert feats.dtype in (torch.float32, torch.float64), f"Expected float type, got {feats.dtype}"

        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.no_grad()
    def compute(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)

        # Compute pairwise distances between real samples (for recall manifold)
        dists_real = self._pairwise_distances(real_feats, real_feats)
        radii_real = dists_real.topk(self.k + 1, largest=False).values[:, -1]

        # Compute pairwise distances between fake samples (for precision manifold)
        dists_fake = self._pairwise_distances(fake_feats, fake_feats)
        radii_fake = dists_fake.topk(self.k + 1, largest=False).values[:, -1]

        # Cross distances: fake-to-real
        dists_cross = self._pairwise_distances(fake_feats, real_feats)
        precision_mask = (dists_cross <= radii_real.unsqueeze(0)).any(dim=1)
        precision = precision_mask.float().mean().item()

        # Cross distances: real-to-fake
        dists_cross_T = dists_cross.t()
        recall_mask = (dists_cross_T <= radii_fake.unsqueeze(0)).any(dim=1)
        recall = recall_mask.float().mean().item()

        return precision, recall

    def _pairwise_distances(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()

    
    
    
############################################
# Image metrics tracker with pFID / rFID
############################################
class ImageMetricsTracker(nn.Module):
    def __init__(self, num_crops=4, crop_size=128, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.lpips.eval()

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        self.prec_recall = PrecisionRecall(k=3, device=self.device).to(self.device)
        self.prec_recall_fid = PrecisionRecallFID(k=3, device=self.device).to(self.device)

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

    def update(self, target, pred):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        def to_float01(x):
            if x.min() >= 0 and x.max() <= 1:
                return x
            elif x.min() >= -1 and x.max() <= 1:
                return (x + 1) / 2
            elif x.dtype == torch.uint8:
                return x.float() / 255
            else:
                raise ValueError(f"Unsupported input range: min {x.min().item()}, max {x.max().item()}")

        def to_uint8(x):
            x01 = to_float01(x)
            return (x01 * 255).round().clamp(0, 255).to(torch.uint8)

        # Ensure uint8 for FID
        real_ims_glb = to_uint8(target)
        fake_ims_glb = to_uint8(pred)

        # Patch-wise FID
        if self.patch_fid:
            croped_real, croped_fake, anchors = [], [], []
            for i in range(real_ims_glb.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(real_ims_glb[0], output_size=(self.crop_size, self.crop_size)))
            for idx, (img_real, img_fake) in enumerate(zip(real_ims_glb, fake_ims_glb)):
                for i in range(self.num_crops):
                    anchor = anchors[idx * self.num_crops + i]
                    croped_real.append(FT.crop(img_real, *anchor))
                    croped_fake.append(FT.crop(img_fake, *anchor))
            real_patches = torch.stack(croped_real)
            fake_patches = torch.stack(croped_fake)
            self.local_fid.update(real_patches, real=True)
            self.local_fid.update(fake_patches, real=False)

        self.global_fid.update(real_ims_glb, real=True)
        self.global_fid.update(fake_ims_glb, real=False)

        self.prec_recall.update(real_ims_glb, real=True)
        self.prec_recall.update(fake_ims_glb, real=False)

        self.prec_recall_fid.update(real_ims_glb, real=True)
        self.prec_recall_fid.update(fake_ims_glb, real=False)

        # Pixel metrics
        pred_norm = to_float01(pred)
        target_norm = to_float01(target)

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
        self.prec_recall.reset()
        self.prec_recall_fid.real_feats = []
        self.prec_recall_fid.fake_feats = []
        if self.patch_fid:
            self.local_fid.reset()

    def aggregate(self):
        precision_val, recall_val = self.prec_recall.compute()
        pFID_val, rFID_val = self.prec_recall_fid.compute_pFID_rFID()

        gfid = self.global_fid.compute().item()
        gfid = max(gfid, 0.0)
        if self.patch_fid:
            lfid = self.local_fid.compute().item()
            lfid = max(lfid, 0.0)
        else:
            lfid = None

        return dict(
            gfid=gfid,
            lfid=lfid,
            precision=max(precision_val, 0.0),
            recall=max(recall_val, 0.0),
            pFID=max(pFID_val, 0.0),
            rFID=max(rFID_val, 0.0),
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item(),
        )




############################################
# Test functions
############################################

def plot_dist_hist(dist, title="Cross distances"):
    dists_flat = dist.cpu().flatten().numpy()
    plt.hist(dists_flat, bins=50, color='skyblue', alpha=0.7)
    plt.title(title)
    plt.xlabel("L2 distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def test_tracker_case(batch_size=64, noise_std=0.3, k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr_metric = PrecisionRecall(k=k, device=device)

    # --- Create random "features" instead of images ---
    feats_real = torch.rand(batch_size, 512, device=device)  # Simulate real features
    feats_fake_noisy = (feats_real + noise_std * torch.randn_like(feats_real)).clamp(0, 1)
    feats_fake_random = torch.rand(batch_size, 512, device=device)

    # --- Case 1: Noisy fake
    pr_metric.reset()
    pr_metric.update(feats_real, real=True)
    pr_metric.update(feats_fake_noisy, real=False)
    precision1, recall1 = pr_metric.compute()
    print(f"Test noisy features → Precision: {precision1:.3f}, Recall: {recall1:.3f}")

    # --- Case 2: Random fake
    pr_metric.reset()
    pr_metric.update(feats_real, real=True)
    pr_metric.update(feats_fake_random, real=False)
    precision2, recall2 = pr_metric.compute()
    print(f"Test random features → Precision: {precision2:.3f}, Recall: {recall2:.3f}")


# ------------------------------------------------
# Test runner
# ------------------------------------------------
if __name__ == "__main__":
    print("Running feature-space Precision-Recall tests...")
    test_tracker_case(batch_size=64, noise_std=0.3, k=3)

    print("\nRunning full ImageMetricsTracker tests including pFID/rFID...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = ImageMetricsTracker(num_crops=4, crop_size=64, device=device)
    tracker.reset()

    batch_size, H, W = 64, 256, 256
    imgs_clean = torch.rand(batch_size, 3, H, W, device=device)

    #####################################
    # Identical images (should be nearly perfect)
    #####################################
    tracker.update(imgs_clean, imgs_clean.clone())
    metrics_same = tracker.aggregate()
    print("\n[IDENTICAL IMAGES]")
    print(metrics_same)
    assert metrics_same["pFID"] < 1.0, "pFID should be very low for identical images"
    assert metrics_same["rFID"] < 1.0, "rFID should be very low for identical images"

    tracker.reset()

    #####################################
    # Noisy images (some degradation)
    #####################################
    imgs_noisy = (imgs_clean + 0.2 * torch.randn_like(imgs_clean)).clamp(0, 1)
    tracker.update(imgs_clean, imgs_noisy)
    metrics_noisy = tracker.aggregate()
    print("\n[NOISY IMAGES]")
    print(metrics_noisy)

    tracker.reset()

    #####################################
    # Completely random images
    #####################################
    imgs_random = torch.rand(batch_size, 3, H, W, device=device)
    tracker.update(imgs_clean, imgs_random)
    metrics_random = tracker.aggregate()
    print("\n[RANDOM IMAGES]")
    print(metrics_random)
    assert metrics_random["pFID"] > metrics_same["pFID"], "pFID should increase for random images"
    assert metrics_random["rFID"] > metrics_same["rFID"], "rFID should increase for random images"

    #####################################
    # Check all required keys
    #####################################
    required_keys = ["gfid", "precision", "recall", "pFID", "rFID", "ssim", "psnr", "mse", "mae", "lpips"]
    for key in required_keys:
        assert key in metrics_random, f"Missing key {key} in metrics dict"

    print("\nAll tests passed, including keys and value sanity checks.")
