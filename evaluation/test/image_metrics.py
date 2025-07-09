
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics


# denorm tensor -- just for plotting
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor
from ldm.helpers import un_normalize_ims  # Convert from [-1, 1] to [0, 255]
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule
from ldm.trainer_rf_vae import TrainerModuleLatentFlow
import os
import sys
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
project_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../../'))
sys.path.append(project_root)


torch.set_float32_matmul_precision('high')


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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
        assert feats.dtype in (
            torch.float32, torch.float64), f"Expected float type, got {feats.dtype}"

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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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

        self.prec_recall = PrecisionRecall(
            k=3, device=self.device).to(self.device)

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
        real_ims_glb = denorm_metrics_tensor(
            target, target_range=(0, 255), dtype='int')
        fake_ims_glb = denorm_metrics_tensor(
            pred, target_range=(0, 255), dtype='int')

        def to_float01(x):
            if x.min() >= 0 and x.max() <= 1:
                return x
            elif x.min() >= -1 and x.max() <= 1:
                return (x + 1) / 2
            elif x.dtype == torch.uint8:
                return x.float() / 255
            else:
                raise ValueError(
                    f"Unsupported input range: min {x.min().item()}, max {x.max().item()}")

        def to_uint8(x):
            x01 = to_float01(x)
            return (x01 * 255).round().clamp(0, 255).to(torch.uint8)

        # Ensureuint8 for FID and Precision-Recall
        real_ims_glb = to_uint8(target)
        fake_ims_glb = to_uint8(pred)

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

        # Normalize pred and target for pixel metrics [0, 1]
        pred_norm = denorm_metrics_tensor(
            pred, target_range=(0, 1), dtype='float')
        target_norm = denorm_metrics_tensor(
            target, target_range=(0, 1), dtype='float')

        # Normalize for pixel-level metrics
        pred_norm = to_float01(pred)
        target_norm = to_float01(target)

        self.ssims.append(self.ssim(pred_norm, target_norm))
        self.psnrs.append(self.psnr(pred_norm, target_norm))
        self.mses.append(torch.mean(
            (pred_norm - target_norm) ** 2, dim=[1, 2, 3]))
        self.maes.append(torch.mean(
            torch.abs(pred_norm - target_norm), dim=[1, 2, 3]))
        self.lpips_scores.append(self.lpips(
            pred_norm * 2 - 1, target_norm * 2 - 1))

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
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item()
        )

    def _check_fid_inputs(self, tensor, name):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a tensor."
        assert tensor.dtype == torch.uint8, f"{name} must be uint8 but got {tensor.dtype}."
        assert tensor.dim() == 4 and tensor.size(
            1) == 3, f"{name} shape must be (N, 3, H, W), got {tensor.shape}."
        assert tensor.min() >= 0 and tensor.max(
        ) <= 255, f"{name} must be in [0, 255], got [{tensor.min()}, {tensor.max()}]."


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
    # Simulate real features
    feats_real = torch.rand(batch_size, 512, device=device)
    feats_fake_noisy = (feats_real + noise_std *
                        torch.randn_like(feats_real)).clamp(0, 1)
    feats_fake_random = torch.rand(batch_size, 512, device=device)

    # --- Case 1: Noisy fake
    pr_metric.reset()
    pr_metric.update(feats_real, real=True)
    pr_metric.update(feats_fake_noisy, real=False)
    precision1, recall1 = pr_metric.compute()
    print(
        f"Test noisy features → Precision: {precision1:.3f}, Recall: {recall1:.3f}")

    # --- Case 2: Random fake
    pr_metric.reset()
    pr_metric.update(feats_real, real=True)
    pr_metric.update(feats_fake_random, real=False)
    precision2, recall2 = pr_metric.compute()
    print(
        f"Test random features → Precision: {precision2:.3f}, Recall: {recall2:.3f}")


# ------------------------------------------------
# Test runner
# ------------------------------------------------
if __name__ == "__main__":
    print("Running improved Precision-Recall tests...")
    test_tracker_case(batch_size=64, noise_std=0.3, k=3)

    print("Running ImageMetricsTracker tests...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use local FID with 4 crops of size 64x64
    tracker = ImageMetricsTracker(num_crops=4, crop_size=64, device=device)
    tracker.reset()

    batch_size, H, W = 64, 256, 256
    imgs_clean = torch.rand(batch_size, 3, H, W, device=device)
    imgs_same = imgs_clean.clone()

    tracker.update(imgs_clean, imgs_same)
    metrics = tracker.aggregate()
    print("Test identical images [0, 1] →", metrics)

    tracker.reset()
    imgs_noisy = (imgs_clean + 0.3 * torch.randn_like(imgs_clean)).clamp(0, 1)
    tracker.update(imgs_clean, imgs_noisy)
    metrics = tracker.aggregate()
    print("Test noisy images →", metrics)

    print("Testing with different images...")
    tracker.reset()
    # Different random images
    imgs_fake = torch.rand(batch_size, 3, H, W, device=device)
    tracker.update(imgs_clean, imgs_fake)
    metrics = tracker.aggregate()
    print("Test with different images →", metrics)

    print("All tests completed successfully")

    # CUDA_VISIBLE_DEVICES=0 python ...
