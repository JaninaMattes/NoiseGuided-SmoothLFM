# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import os
import sys
import gc

from tqdm import tqdm

import torch
import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from lightning import seed_everything

import numpy as np
from typing import Tuple

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List

from matplotlib import pyplot as plt
from matplotlib import rcParams

# helper
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from pytorch_fid.inception import InceptionV3


# Jutils
from jutils import freeze


# Setup project root for import resolution
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
)
sys.path.append(project_root)

from ldm.trainer_rf_vae import TrainerModuleLatentFlow
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

from data_processing.tools.norm import (
    denorm_metrics_tensor,
    denorm_tensor,
)  # denorm tensor -- just for plotting


torch.set_float32_matmul_precision("high")


#########################################################
#                    Metric Tracker Classes             #
#########################################################
class SmoothnessMetricsTracker(nn.Module):
    """
    Calculates smoothness metrics PPL and ISTD from a sequence of generated images.
    Based on:
    [0] PPL: "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
    [1] Smooth Diffusion: "Crafting Smooth Latent Spaces in Diffusion Models" (Guo et al., 2024)
    """

    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lpips = LPIPS(net_type="vgg").to(self.device)
        self.lpips.eval()
        self.reset()

    def reset(self):
        self.ppls = []
        self.istds = []

    @torch.no_grad()
    def update(self, interpolated_imgs_batch):
        assert interpolated_imgs_batch.dim() == 5, (
            f"Expected 5D tensor, got {interpolated_imgs_batch.dim()}D"
        )
        B, T, C, H, W = interpolated_imgs_batch.shape
        assert T > 1, "Each sequence must contain at least 2 images."

        # Normalize only if needed (i.e., not already in [-1, 1])
        min_val, max_val = interpolated_imgs_batch.min(), interpolated_imgs_batch.max()
        if min_val < -1.01 or max_val > 1.01:
            # Rescale to [-1, 1] dynamically per sequence
            min_per_seq = interpolated_imgs_batch.amin(dim=(2, 3, 4), keepdim=True)
            max_per_seq = interpolated_imgs_batch.amax(dim=(2, 3, 4), keepdim=True)
            denom = (max_per_seq - min_per_seq).clamp(min=1e-5)
            interpolated_imgs_batch = (
                2 * (interpolated_imgs_batch - min_per_seq) / denom - 1
            )

        batch = interpolated_imgs_batch.to(self.device)

        for i in range(B):
            sequence = batch[i]  # (T, C, H, W)
            dists = []

            with torch.amp.autocast("cuda"):
                for t in range(T - 1):
                    d = self.lpips(
                        sequence[t].unsqueeze(0), sequence[t + 1].unsqueeze(0)
                    ).item()
                    dists.append(d)

            if len(dists) == 0:
                print("[WARN] No valid LPIPS distances computed.")
                continue

            self.ppls.append(float(torch.tensor(dists).mean()))
            self.istds.append(float(torch.tensor(dists).std()))

        print(f"[INFO] Processed {B} sequences for smoothness metrics.")

    @torch.no_grad()
    def aggregate(self):
        if not self.ppls:
            print("Warning: No data in tracker. Call update() before aggregate().")
            return {"ppl": float("nan"), "istd": float("nan")}

        mean_ppl = torch.tensor(self.ppls).mean().item()
        mean_istd = torch.tensor(self.istds).mean().item()

        return {"ppl": mean_ppl, "istd": mean_istd}


############################################
#   FID class using InceptionV3
############################################
class PrecisionRecallFID(nn.Module):
    def __init__(self, k=3, device=None):
        super().__init__()
        self.k = k
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        x_norm = (x**2).sum(dim=1).unsqueeze(1)
        y_norm = (y**2).sum(dim=1).unsqueeze(0)
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
            covmean = linalg.sqrtm(
                (sigma1.cpu().numpy() + offset) @ (sigma2.cpu().numpy() + offset)
            )

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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        assert feats.dtype in (torch.float32, torch.float64), (
            f"Expected float type, got {feats.dtype}"
        )

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
        x_norm = (x**2).sum(dim=1).unsqueeze(1)
        y_norm = (y**2).sum(dim=1).unsqueeze(0)
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
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type="vgg").to(self.device)
        self.lpips.eval()

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True,
        ).to(self.device)

        self.prec_recall = PrecisionRecall(k=3, device=self.device).to(self.device)
        self.prec_recall_fid = PrecisionRecallFID(k=3, device=self.device).to(
            self.device
        )

        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[ImageMetricsTracker] Evaluating using patch-wise FID")
            self.local_fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True,
            ).to(self.device)

        self.num_crops = num_crops
        self.crop_size = crop_size

        self.reset()

    def update(self, target, pred):
        assert pred.shape == target.shape, (
            f"Shape mismatch: {pred.shape} vs {target.shape}"
        )

        # Convert to [0, 255] uint8 for FID and PR metrics
        real_ims_glb = denorm_metrics_tensor(target, target_range=(0, 255), dtype="int")
        fake_ims_glb = denorm_metrics_tensor(pred, target_range=(0, 255), dtype="int")

        def to_float01(x):
            if x.min() >= 0 and x.max() <= 1:
                return x
            elif x.min() >= -1 and x.max() <= 1:
                return (x + 1) / 2
            elif x.dtype == torch.uint8:
                return x.float() / 255
            else:
                raise ValueError(
                    f"Unsupported input range: min {x.min().item()}, max {x.max().item()}"
                )

        # Local (patch) FID
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []

            for i in range(real_ims_glb.shape[0] * self.num_crops):
                anchors.append(
                    transforms.RandomCrop.get_params(
                        real_ims_glb[0], output_size=(self.crop_size, self.crop_size)
                    )
                )

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
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype="float")
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype="float")

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
            lpips=torch.stack(self.lpips_scores).mean().item(),
        )

    def _check_fid_inputs(self, tensor, name):
        assert isinstance(tensor, torch.Tensor), f"{name} must be a tensor."
        assert tensor.dtype == torch.uint8, (
            f"{name} must be uint8 but got {tensor.dtype}."
        )
        assert tensor.dim() == 4 and tensor.size(1) == 3, (
            f"{name} shape must be (N, 3, H, W), got {tensor.shape}."
        )
        assert tensor.min() >= 0 and tensor.max() <= 255, (
            f"{name} must be in [0, 255], got [{tensor.min()}, {tensor.max()}]."
        )


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
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        assert feats.dtype in (torch.float32, torch.float64), (
            f"Expected float type, got {feats.dtype}"
        )

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
        x_norm = (x**2).sum(dim=1).unsqueeze(1)
        y_norm = (y**2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()


############################################
# Image metrics tracker with pFID / rFID
############################################
class ImageMetricsTracker(nn.Module):
    def __init__(self, num_crops=4, crop_size=128, device=None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.ssim = SSIM(data_range=1.0).to(self.device)
        self.psnr = PSNR(data_range=1.0).to(self.device)
        self.mse = nn.MSELoss()

        self.lpips = LPIPS(net_type="vgg").to(self.device)
        self.lpips.eval()

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True,
        ).to(self.device)

        self.prec_recall = PrecisionRecall(k=3, device=self.device).to(self.device)
        self.prec_recall_fid = PrecisionRecallFID(k=3, device=self.device).to(
            self.device
        )

        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[ImageMetricsTracker] Evaluating using patch-wise FID")
            self.local_fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True,
            ).to(self.device)

        self.num_crops = num_crops
        self.crop_size = crop_size

        self.reset()

    def update(self, target, pred):
        assert pred.shape == target.shape, (
            f"Shape mismatch: {pred.shape} vs {target.shape}"
        )

        # Convert to [0, 255] uint8 for FID and PR metrics
        real_ims_glb = denorm_metrics_tensor(target, target_range=(0, 255), dtype="int")
        fake_ims_glb = denorm_metrics_tensor(pred, target_range=(0, 255), dtype="int")

        # Patch-wise FID
        if self.patch_fid:
            croped_real, croped_fake, anchors = [], [], []
            for i in range(real_ims_glb.shape[0] * self.num_crops):
                anchors.append(
                    transforms.RandomCrop.get_params(
                        real_ims_glb[0], output_size=(self.crop_size, self.crop_size)
                    )
                )
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

        # Normalize pred and target for pixel metrics [0, 1]
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype="float")
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype="float")

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
        raise ValueError(
            "[interpolate_vectors] alpha_vals is empty. Ensure num_interpolations > 0."
        )

    if mode == "linear":
        return torch.stack([(1 - alpha) * z1 + alpha * z2 for alpha in alpha_vals])
    elif mode == "slerp":
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        dot = torch.dot(z1_norm, z2_norm).clamp(-1.0, 1.0)

        if torch.abs(dot) > dot_threshold:
            return torch.stack([torch.lerp(z1, z2, alpha) for alpha in alpha_vals])
        else:
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            return torch.stack(
                [
                    torch.sin((1.0 - alpha) * omega) / sin_omega * z1_norm
                    + torch.sin(alpha * omega) / sin_omega * z2_norm
                    for alpha in alpha_vals
                ]
            )
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
    fm_module,
    smoothness_tracker,
    num_pairs=10,
    num_interpolations=16,
    cfg_scale=1.0,
    ccfg_scale=0.0,
    sample_kwargs=None,
    use_labels=False,
    num_classes=1000,
    device=None,
    interp_type="linear",
    upscale_to=128,
    plot_samples=False,
    save_dir=None,
    title=None,
    gt_border=0,
    line_width=5,
):
    """
    Interpolates between latent pairs and evaluates smoothness (PPL + ISTD) for each sequence.
    Optionally plots a grid of interpolated sequences once, per class-pair.
    """
    device = device or fm_module.device
    alpha_lin_space = torch.linspace(0, 1, num_interpolations).to(device)

    all_sequences = []
    generated_rows = []
    smoothness_tracker.reset()

    for i in range(num_pairs):
        # === Step 1: Get latents and encode context ===
        x1 = cls1_latents[i].unsqueeze(0).to(device)
        x2 = cls2_latents[i].unsqueeze(0).to(device)

        z1 = fm_module.encode_third_stage(x1).squeeze(0).float()
        z2 = fm_module.encode_third_stage(x2).squeeze(0).float()

        interp_context = interpolate_vectors(
            z1, z2, alpha_lin_space, mode=interp_type
        ).to(device)

        # === Step 2: Generate interpolated images ===
        B, C, H, W = num_interpolations, *x1.shape[1:]
        random_x = torch.randn(B, C, H, W, device=device)

        uc_context = torch.zeros_like(interp_context)
        uc_cond = (
            torch.full((B,), num_classes, device=device, dtype=torch.long)
            if use_labels
            else None
        )
        labels = (
            torch.cat(
                [
                    torch.full((B // 2,), cls1_label, dtype=torch.long),
                    torch.full((B - B // 2,), cls2_label, dtype=torch.long),
                ]
            ).to(device)
            if use_labels
            else None
        )

        kwargs = sample_kwargs or {}
        kwargs.update(
            {
                "num_steps": kwargs.get("num_steps", 50),
                "progress": False,
                "context": interp_context,
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "uc_cond_context": uc_context,
                "uc_cond": uc_cond,
                "y": labels,
            }
        )

        generated = fm_module.model.generate(x=random_x, **kwargs)
        decoded_interpolation = fm_module.decode_first_stage(generated)

        # === Step 3: Smoothness Eval ===
        all_sequences.append(decoded_interpolation.unsqueeze(0))

        # === Step 4: Prepare images for grid ===
        if plot_samples:
            # Denorm is fine for visualization, but not for metrics
            row_images = denorm_tensor(decoded_interpolation * 0.5 + 0.5).detach().cpu()
            row_images = torch.stack(
                [TF.resize(img, [upscale_to, upscale_to]) for img in row_images]
            )

            real_start = denorm_tensor(cls1_images[i].unsqueeze(0))[0].detach().cpu()
            real_start = TF.resize(real_start, [upscale_to, upscale_to])
            real_end = denorm_tensor(cls2_images[i].unsqueeze(0))[0].detach().cpu()
            real_end = TF.resize(real_end, [upscale_to, upscale_to])

            # Pad real_start and real_end to match the grid size
            real_start_padded = F.pad(
                real_start, pad=[0, gt_border, 0, 0], mode="constant", value=0
            )
            real_end_padded = F.pad(
                real_end, pad=[gt_border, 0, 0, 0], mode="constant", value=0
            )

            pad_left = gt_border // 2
            pad_right = gt_border - pad_left
            row_images_padded = torch.stack(
                [
                    F.pad(
                        img, pad=[pad_left, pad_right, 0, 0], mode="constant", value=0
                    )
                    for img in row_images
                ]
            )

            full_row = torch.cat(
                [
                    real_start_padded.unsqueeze(0),
                    row_images_padded,
                    real_end_padded.unsqueeze(0),
                ],
                dim=0,
            )
            generated_rows.append(full_row)

    if plot_samples and save_dir is not None and generated_rows:
        all_rows = torch.cat(
            generated_rows, dim=0
        )  # shape: (num_pairs * (T+2), C, H, W)
        nrow = num_interpolations + 2
        grid = make_grid(all_rows, nrow=nrow, padding=0)
        grid_np = grid.permute(1, 2, 0).numpy()

        rcParams.update({"font.size": 14, "font.family": "DejaVu Sans"})
        fig, ax = plt.subplots(figsize=(grid.shape[2] / 50, grid.shape[1] / 50))
        ax.imshow(grid_np)
        ax.axis("off")

        # === FIXED: Only draw vertical lines after first and before last ===
        img_width = grid_np.shape[1] / nrow
        ax.axvline(x=img_width, color="black", linewidth=line_width)  # After Real A
        ax.axvline(
            x=(nrow - 1) * img_width, color="black", linewidth=line_width
        )  # Before Real B

        xtick_positions = [(i + 0.5) * img_width for i in range(nrow)]
        xtick_labels = (
            ["Real A"]
            + [f"{alpha:.2f}" for alpha in alpha_lin_space.tolist()]
            + ["Real B"]
        )
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45)

        ytick_positions = [
            (i + 0.5) * (grid_np.shape[0] / len(generated_rows))
            for i in range(len(generated_rows))
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(
            [f"Pair {i}" for i in range(len(generated_rows))], fontsize=10
        )

        title = title or f"Latent Interpolation Grid ({interp_type})"
        ax.set_title(title, fontsize=14)

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.tight_layout()

        save_dir = os.path.join(
            save_dir, f"{interp_type}_cls1_{cls1_label}_cls2_{cls2_label}.png"
        )
        plt.savefig(save_dir, bbox_inches="tight", dpi=500)
        print(f"[INFO] Saved interpolation grid to: {save_dir}")

        plt.show()
        plt.close(fig)

        del grid, grid_np, fig, ax, all_rows
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
import torchvision.transforms.functional as TF


def generate_samples(
    fm_module,
    images,
    xt_latent,
    labels=None,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    num_steps=50,
    num_classes=1000,
    max_samples=None,
    device=None,
    denorm_fn=None,  # If needed later
    plot_samples=False,
    nrow=4,
    title="Generated Samples",
    save_path=None,
    resize_to=128,
):
    device = device or fm_module.device

    # Crop batch
    if max_samples is not None:
        images = images[:max_samples]
        xt_latent = xt_latent[:max_samples]
        if labels is not None:
            labels = labels[:max_samples]

    # Move tensors to device
    images = images.to(device)
    xt_latent = xt_latent.to(device)
    if labels is not None:
        labels = labels.to(device).squeeze()

    with torch.no_grad():
        context = fm_module.encode_third_stage(xt_latent)
        z = torch.randn_like(xt_latent)

        uc_context = torch.zeros_like(context)
        uc_label = torch.full(
            (xt_latent.size(0),), num_classes, device=device, dtype=torch.long
        )

        sample_kwargs = {
            "num_steps": num_steps,
            "progress": False,
            "context": context,
            "y": labels,
            "cfg_scale": cfg_scale,
            "ccfg_scale": ccfg_scale,
            "uc_cond_context": uc_context,
            "uc_cond": uc_label,
        }

        generated = fm_module.model.generate(x=z, **sample_kwargs)
        fake_images = fm_module.decode_first_stage(generated)
        real_images = images  # Already unnormalized

    # Plotting (optional)
    # Denorm is okay for visualization, but not for metrics
    if plot_samples:
        real_images_ = denorm_tensor(real_images).detach().cpu()
        fake_images_ = denorm_tensor(fake_images).detach().cpu()
        real_images_resized = TF.resize(real_images_, [resize_to, resize_to])
        fake_images_resized = TF.resize(fake_images_, [resize_to, resize_to])

        interleaved = []
        for real, fake in zip(real_images_resized, fake_images_resized):
            interleaved.extend([real, fake])

        grid = make_grid(torch.stack(interleaved), nrow=nrow, padding=0)

        rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
        fig, ax = plt.subplots(figsize=(grid.shape[2] / 50, grid.shape[1] / 50))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
        ax.set_title(title, fontsize=14)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"[INFO] Saved sample plot to {save_path}")

        plt.show()
        plt.close(fig)

        del (
            fig,
            grid,
            fake_images_,
            real_images_,
            real_images_resized,
            fake_images_resized,
            interleaved,
        )
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    # Clean up
    del images, xt_latent, context, z, uc_context, uc_label, generated
    torch.cuda.empty_cache()
    gc.collect()

    return fake_images.detach().cpu(), real_images.detach().cpu()


#########################################################
#                    Collect Samples                    #
#########################################################
def collect_samples(
    data,
    class_labels: List[int],
    source_timestep: float,
    samples_per_class: int = 10,
    group_name="validation",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collects a fixed number of samples per class from a validation dataloader.
    Returns latents, labels, and corresponding images.
    """
    collected_latents = defaultdict(list)
    collected_images = defaultdict(list)

    # Dataloader
    if group_name == "validation":
        val_loader = data.val_dataloader()
    else:
        val_loader = data.test_dataloader()

    val_loader.shuffle = True
    latent_key = f"latents_{source_timestep:.2f}"

    for batch_idx, batch in enumerate(val_loader):
        latents = batch[latent_key].detach().cpu()
        labels = batch["label"].view(-1).detach().cpu()
        images = batch["image"].detach().cpu()

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

        if all(
            len(collected_latents[label]) >= samples_per_class for label in class_labels
        ):
            print(
                f"[INFO] Collected enough samples for all classes after {batch_idx + 1} batches."
            )
            break

    all_latents, all_labels, all_images = [], [], []
    for label in class_labels:
        latents_list = collected_latents[label]
        images_list = collected_images[label]

        if len(latents_list) < samples_per_class:
            raise ValueError(
                f"Not enough samples collected for class {label}: got {len(latents_list)}"
            )

        stacked_latents = torch.stack(latents_list[:samples_per_class], dim=0)
        stacked_images = torch.stack(images_list[:samples_per_class], dim=0)
        label_tensor = torch.full((samples_per_class,), label, dtype=torch.long)

        all_latents.append(stacked_latents)
        all_labels.append(label_tensor)
        all_images.append(stacked_images)

    return (
        torch.cat(all_latents, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_images, dim=0),
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
@torch.no_grad()
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
    max_samples=10000,
    num_steps=50,
    num_classes=1000,
    num_crops=4,
    crop_size=64,
    device=None,
    plot_every_n_batches=1000,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    fm_module = TrainerModuleLatentFlow.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    fm_module.eval().to(device)
    freeze(fm_module.model)

    num_params = sum(p.numel() for p in fm_module.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(
        f"[INFO] Running evaluation for group: {group}, model: {model_name}, dataset: {dataset_name}"
    )

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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # CSV setup
    csv_path = base_results_dir / f"{model_name}_metrics.csv"
    with open(csv_path, "w") as csv_file:
        csv_file.write(
            "evaluation_type,model_name,dataset,CFG,CCFG,gFID,lFID,pFID,rFID,SSIM,PSNR,MSE,MAE,LPIPS,PPL,ISTD,time_per_image_ms,interpolation_pair\n"
        )

        ##############################
        # PART 1: Image Quality
        ##############################
        print("\n--- Part 1: Image Quality Evaluation ---")
        tracker = ImageMetricsTracker(
            num_crops=num_crops, crop_size=crop_size, device=device
        )

        dataloader = get_dataloader_by_group(data, group)
        dataloader.shuffle = True

        for cfg_scale, ccfg_scale in zip(cfg_scales, ccfg_scales):
            print(f"[INFO] Evaluating CFG={cfg_scale}, CCFG={ccfg_scale}")
            tracker.reset()

            count = 0
            total_time_ms = 0.0
            total_images = 0

            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                if batch["image"].size(0) < batch_size:
                    print(
                        f"[WARN] Skipping batch {batch_idx} with size {batch['image'].size(0)} < {batch_size}"
                    )
                    continue

                if count >= max_samples:
                    print(f"[INFO] Reached max sample limit: {max_samples}")
                    break

                images = batch["image"]
                xt = batch[f"latents_{source_timestep:.2f}"]
                labels = batch["label"]

                plot_now = batch_idx % plot_every_n_batches == 0
                plot_path = str(
                    base_results_dir
                    / f"batch_{batch_idx}_cfg{cfg_scale}_ccfg{ccfg_scale}.png"
                )
                title_str = f"Generated Samples (CFG={cfg_scale}, CCFG={ccfg_scale})"

                # ==== Start: Measure time ====
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()

                fake_imgs, real_imgs = generate_samples(
                    fm_module=fm_module,
                    images=images,
                    xt_latent=xt,
                    labels=labels,
                    cfg_scale=cfg_scale,
                    ccfg_scale=ccfg_scale,
                    num_steps=num_steps,
                    num_classes=num_classes,
                    device=device,
                    plot_samples=plot_now,
                    save_path=plot_path if plot_now else None,
                    title=title_str,
                )

                # ==== End: Measure time ====
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_time.elapsed_time(end_time)

                total_time_ms += elapsed_time_ms
                total_images += fake_imgs.size(0)

                # Update metrics tracker
                tracker.update(real_imgs.to(device), fake_imgs.to(device))
                count += real_imgs.size(0)

                del real_imgs, fake_imgs, images, xt, labels, start_time, end_time
                torch.cuda.empty_cache()
                gc.collect()

            # ==== Final: Aggregate metrics ====
            # After 50k samples for CFG collected
            average_time_per_image = (
                total_time_ms / total_images if total_images > 0 else 0.0
            )

            metrics = tracker.aggregate()
            print(
                f"[INFO] CFG={cfg_scale}, CCFG={ccfg_scale} → Avg Time: {average_time_per_image:.4f} ms, gFID: {metrics['gfid']:.4f}, lFID: {metrics['lfid']:.4f}, SSIM: {metrics['ssim']:.4f}, "
                f"MSE: {metrics['mse']:.4f}, LPIPS: {metrics['lpips']:.4f}, PSNR: {metrics['psnr']:.4f}"
            )

            # Write also runtime to CSV
            csv_file.write(
                f"image_quality,{model_name},{dataset_name},{cfg_scale},{ccfg_scale},"
                f"{metrics['gfid']:.6f},{metrics['lfid'] if metrics['lfid'] is not None else float('nan'):.6f},"
                f"{metrics['pFID']:.6f},{metrics['rFID']:.6f},"
                f"{metrics['ssim']:.6f},{metrics['psnr']:.6f},{metrics['mse']:.6f},{metrics['mae']:.6f},{metrics['lpips']:.6f},"
                f"nan,nan,{average_time_per_image:.4f},nan\n"
            )
            csv_file.flush()

            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()

        ##############################
        # PART 2: Smoothness Metrics
        ##############################

        print("\n--- Part 2: Interpolation Smoothness Evaluation ---")
        smoothness_tracker = SmoothnessMetricsTracker(device=device)

        all_classes = {cls for pair in interpolation_dict.values() for cls in pair}
        latents, labels, images = collect_samples(
            data=data,
            class_labels=list(all_classes),
            source_timestep=source_timestep,
            samples_per_class=max(samples_per_class, num_pairs),
            group_name=group,
        )

        global_ppl = 0.0
        global_istd = 0.0

        count_interp = 0
        for interp_name, (cls_a, cls_b) in interpolation_dict.items():
            # reset per grid pair
            smoothness_tracker.reset()

            cls1_latents = latents[labels == cls_a][:num_pairs]
            cls2_latents = latents[labels == cls_b][:num_pairs]
            cls1_images = images[labels == cls_a][:num_pairs]
            cls2_images = images[labels == cls_b][:num_pairs]

            print(f"[INFO] Interpolating: {interp_name} ({cls_a} → {cls_b})")
            plot_now = count_interp % 2 == 0

            interpolate_latents_with_smoothness_eval(
                cls1_latents=cls1_latents,
                cls2_latents=cls2_latents,
                cls1_images=cls1_images,
                cls2_images=cls2_images,
                cls1_label=cls_a,
                cls2_label=cls_b,
                fm_module=fm_module,
                smoothness_tracker=smoothness_tracker,
                num_pairs=num_pairs,
                num_interpolations=num_interpolations,
                cfg_scale=2.0,
                ccfg_scale=1.0,
                sample_kwargs={"num_steps": num_steps},
                use_labels=False,
                num_classes=num_classes,
                device=device,
                interp_type="linear",
                plot_samples=plot_now,
                save_dir=base_results_dir if plot_now else None,
                title=f"Synthesized Interpolation {interp_name} ({cls_a} → {cls_b})",
            )
            count_interp += 1

            out = smoothness_tracker.aggregate()
            mean_ppl, mean_istd = out["ppl"], out["istd"]
            print(f"[INFO] Number of Run Interpolations: {count_interp}")
            print(f"[INFO] {interp_name} → PPL: {mean_ppl:.4f}, ISTD: {mean_istd:.4f}")

            global_ppl += mean_ppl
            global_istd += mean_istd

            # Write to CSV
            csv_file.write(
                f"smoothness,{model_name},{dataset_name},nan,nan,nan,nan,nan,nan,nan,"
                f"{mean_ppl:.6f},{mean_istd:.6f},nan,{interp_name}\n"
            )
            csv_file.flush()

            torch.cuda.empty_cache()
            gc.collect()

        avg_ppl = global_ppl / len(interpolation_dict)
        avg_istd = global_istd / len(interpolation_dict)
        print(
            f"\n[INFO] Global Average PPL: {avg_ppl:.4f}, Global Average ISTD: {avg_istd:.4f}"
        )
        # Write global averages to CSV
        csv_file.write(
            f"smoothness_average,{model_name},{dataset_name},nan,nan,nan,nan,nan,nan,nan,"
            f"{avg_ppl:.6f},{avg_istd:.6f},nan,ALL_PAIRS\n"
        )
        csv_file.flush()

        print(
            f"\n[✓] Interpolation complete. Avg PPL: {avg_ppl:.4f}, Avg ISTD: {avg_istd:.4f}"
        )


#########################################################
#                         END RUN                       #
#########################################################


if __name__ == "__main__":
    #####################################
    # Evaluation Setup
    #####################################
    source_timestep = 0.20
    target_timestep = 1.00
    beta = 0.1
    dataset_name = "imagenet256-dataset-T000006"
    test_dataset_name = "imagenet256-testset-T151412"
    group = "validation"  # or "test"
    baseline = source_timestep == 0.50 and target_timestep == 0.50

    #####################################
    # Device + Seed Setup
    #####################################
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()

    #####################################
    # Model Paths for SiT-XL-2
    #####################################

    # beta: 0.1
    DiTSXL_Beta05x05x_01b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt"  ### (Baseline)
    DiTSXL_Beta00x10x_01b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt"  ### Done
    DITSXL_BETA02x10x_01b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_2.0b/BetaVAE-B-2/V0/2025-07-10/30912/checkpoints/last.ckpt"  ### Done
    DITSXL_BETA05x10x_01b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt"

    # beta: 1.0
    DITSXL_Beta05x05x_1b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt"
    DITSXL_Beta02x10x_1b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt"
    DITSXL_Beta05x10x_1b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt"

    # beta: 5.0
    DiTSXL_Beta05x05x_5b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt"
    DiTSXL_Beta02x10x_5b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt"
    DiTSXL_Beta05x10x_5b = "./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt"

    #####################################
    # Dataset & Evaluation Parameters
    #####################################
    checkpoint = DITSXL_BETA02x10x_01b
    test_data_path = "./dataset/processed/testset-256/imagenet256-testset-T190319.hdf5"
    validation_data_path = "./dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5"  # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'
    project_name = (
        "CFM_Quantitative_Eval_Baseline" if baseline else "CFM_Quantitative_Eval"
    )
    model_name = (
        f"Beta-VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"
    )

    #####################################
    # Interpolation Class Pairs
    #####################################
    interpolation_dict = {
        "admiral_to_cabbage_butterly": [321, 324],
        "monarch_to_admiral_butterly": [323, 321],
        "siamese_to_persian_cat": [284, 283],
        "red_panda_to_giant_panda": [387, 388],
        "pembroke_corgi_to_cardigan_corgi": [263, 264],
        "husky_to_doberman": [250, 236],
        "lion_to_tiger": [291, 292],
        "grey_to_white_wolf": [269, 270],
        "horse_to_zebra": [339, 340],
        "camel_to_impala": [339, 353],
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
    torch.cuda.manual_seed(2025)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    samples_per_class = 10  # Number of samples per class for interpolation
    num_pairs = 10  # Number of pairs to interpolate between classes
    num_interpolations = 16  # Number of interpolation steps between each pair
    max_samples = 1000  # Maximum number of samples to process in total (for FID, etc.)
    batch_size = 16  # Batch size for evaluation
    cfg_scales = [1.0, 3.0, 5.0, 7.0, 9.0]  # Context-Conditional CFG scales [1.0,]
    ccfg_scales = [1.0, 1.0, 1.0, 1.0, 1.0]  # Class-conditional CFG scales [1.0,]

    #####################################
    # Run Evaluation
    #####################################
    run_quant_eval(
        checkpoint=checkpoint,
        data_path=validation_data_path if group == "validation" else test_data_path,
        interpolation_dict=interpolation_dict,
        project_name=project_name,
        model_name=model_name,
        group=group,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        beta=beta,
        samples_per_class=samples_per_class,
        num_pairs=num_pairs,
        num_interpolations=num_interpolations,
        cfg_scales=cfg_scales,
        ccfg_scales=ccfg_scales,
        batch_size=batch_size,
        dataset_name=dataset_name,
        results_root="results",
        max_samples=max_samples,  # 50 k for FID (50000 samples)
        num_steps=50,
        num_classes=1000,
        num_crops=4,  # Set to 0 for no crops, or >0 for cropping (more local features)
        crop_size=64,
    )


# CUDA_VISIBLE_DEVICES=2 python ...
