
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
# - https://github.com/clovaai/generative-evaluation-prdc/blob/master/README.md

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
import torchvision.transforms.functional as TF
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


from prdc import compute_prdc
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

        pFID = self._compute_fid(real_feats, fake_in_real)
        rFID = self._compute_fid(fake_feats, real_in_fake)

        torch.cuda.empty_cache()
        return pFID, rFID

    @torch.no_grad()
    def compute_prdc_metrics(self):
        real_feats = torch.cat(self.real_feats, dim=0).cpu().numpy()
        fake_feats = torch.cat(self.fake_feats, dim=0).cpu().numpy()

        metrics = compute_prdc(
            real_features=real_feats,
            fake_features=fake_feats,
            nearest_k=self.k
        )

        return metrics

    def _pairwise(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()

    @torch.no_grad()
    def _compute_fid(self, feats1, feats2):
        mu1 = feats1.mean(dim=0)
        mu2 = feats2.mean(dim=0)

        sigma1 = self._cov(feats1)
        sigma2 = self._cov(feats2)

        diff = mu1 - mu2

        cov_prod = sigma1 @ sigma2
        eigvals, eigvecs = torch.linalg.eigh(cov_prod)
        covmean = eigvecs @ torch.diag(eigvals.clamp(min=0).sqrt()) @ eigvecs.T

        fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    @torch.no_grad()
    def _cov(self, feats):
        n = feats.shape[0]
        mean = feats.mean(dim=0, keepdim=True)
        feats_centered = feats - mean
        cov = feats_centered.T @ feats_centered / (n - 1)
        return cov





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
        if feats.dtype == torch.uint8:
            feats = feats.float() / 255.0
        feats = feats.view(feats.size(0), -1)
        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.no_grad()
    def compute(self):
        real_feats = torch.cat(self.real_feats, dim=0).to(self.device)
        fake_feats = torch.cat(self.fake_feats, dim=0).to(self.device)

        dists_real = self._pairwise_distances(real_feats, real_feats)
        radii_real = dists_real.topk(self.k + 1, largest=False).values[:, -1]

        dists_fake = self._pairwise_distances(fake_feats, fake_feats)
        radii_fake = dists_fake.topk(self.k + 1, largest=False).values[:, -1]

        dists_cross = self._pairwise_distances(fake_feats, real_feats)

        precision_mask = (dists_cross <= radii_real.unsqueeze(0)).any(dim=1)
        precision = precision_mask.float().mean().item()

        recall_mask = (dists_cross.t() <= radii_fake.unsqueeze(0)).any(dim=1)
        recall = recall_mask.float().mean().item()

        del real_feats, fake_feats, dists_real, dists_fake, dists_cross, radii_real, radii_fake
        torch.cuda.empty_cache()

        return precision, recall

    @torch.no_grad()
    def _pairwise_distances(self, x, y):
        x_norm = x.pow(2).sum(1).unsqueeze(1)
        y_norm = y.pow(2).sum(1).unsqueeze(0)
        dist = x_norm + y_norm - 2 * x @ y.t()
        return dist.clamp(min=0).sqrt()


    
    
    
############################################
# Image metrics tracker with pFID / rFID
############################################
class FIDMetricsTracker(nn.Module):
    def __init__(self, num_crops=4, crop_size=128, k=3, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        self.prec_recall = PrecisionRecall(k=k, device=self.device).to(self.device)
        self.prec_recall_fid = PrecisionRecallFID(k=k, device=self.device).to(self.device)

        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[FIDMetricsTracker] Evaluating using patch-wise FID")
            self.local_fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True
            ).to(self.device)

        self.num_crops = num_crops
        self.crop_size = crop_size

        self.reset()

    def reset(self):
        self.global_fid.reset()
        self.prec_recall.reset()
        self.prec_recall_fid.real_feats = []
        self.prec_recall_fid.fake_feats = []
        if self.patch_fid:
            self.local_fid.reset()
            
    @torch.no_grad()
    def update(self, target, pred):

        # Convert to [0, 255] uint8 for FID and PR metrics
        real_ims = denorm_metrics_tensor(target, target_range=(0, 255), dtype='int').to(self.device)
        fake_ims = denorm_metrics_tensor(pred, target_range=(0, 255), dtype='int').to(self.device)

        # Patch-wise FID
        if self.patch_fid:
            cropped_real, cropped_fake, anchors = [], [], []
            for i in range(real_ims.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(real_ims[0], output_size=(self.crop_size, self.crop_size)))
            for idx, (img_real, img_fake) in enumerate(zip(real_ims, fake_ims)):
                for i in range(self.num_crops):
                    anchor = anchors[idx * self.num_crops + i]
                    cropped_real.append(TF.crop(img_real, *anchor))
                    cropped_fake.append(TF.crop(img_fake, *anchor))
            real_patches = torch.stack(cropped_real)
            fake_patches = torch.stack(cropped_fake)
            self.local_fid.update(real_patches, real=True)
            self.local_fid.update(fake_patches, real=False)

        self.global_fid.update(real_ims, real=True)
        self.global_fid.update(fake_ims, real=False)

        self.prec_recall.update(real_ims, real=True)
        self.prec_recall.update(fake_ims, real=False)

        self.prec_recall_fid.update(real_ims, real=True)
        self.prec_recall_fid.update(fake_ims, real=False)

        # Clean up memory
        del real_ims, fake_ims
        torch.cuda.empty_cache()

    @torch.no_grad()
    def aggregate(self):
        precision_val, recall_val = self.prec_recall.compute()
        pFID_val, rFID_val = self.prec_recall_fid.compute_pFID_rFID()
        prdc = self.prec_recall_fid.compute_prdc_metrics()

        gfid = self.global_fid.compute().item()
        gfid = max(gfid, 0.0)
        lfid = self.local_fid.compute().item() if self.patch_fid else None

        return dict(
            gfid=gfid,
            lfid=max(lfid, 0.0) if lfid is not None else None,
            precision=max(precision_val, 0.0),
            recall=max(recall_val, 0.0),
            pFID=max(pFID_val, 0.0),
            rFID=max(rFID_val, 0.0),
            prdc_precision=prdc['precision'],
            prdc_recall=prdc['recall'],
            prdc_density=prdc['density'],
            prdc_coverage=prdc['coverage'],
        )



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

        self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.lpips.eval()

        self.reset()

    def reset(self):
        self.ssims = []
        self.psnrs = []
        self.mses = []
        self.maes = []
        self.lpips_scores = []

    @torch.no_grad()
    def update(self, target, pred):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Normalize pred and target for pixel metrics [0, 1]
        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float').to(self.device)
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float').to(self.device)

        self.ssims.append(self.ssim(pred_norm, target_norm).detach().cpu())
        self.psnrs.append(self.psnr(pred_norm, target_norm).detach().cpu())
        self.mses.append(torch.mean((pred_norm - target_norm) ** 2, dim=[1, 2, 3]).detach().cpu())
        self.maes.append(torch.mean(torch.abs(pred_norm - target_norm), dim=[1, 2, 3]).detach().cpu())
        self.lpips_scores.append(self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1).detach().cpu()) # Expect input in [-1, 1] for LPIPS
        
        # Clean up memory
        del pred_norm, target_norm
        torch.cuda.empty_cache()

    def aggregate(self):
        
        return dict(
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item(),
        )



class CombinedMetricsTracker(nn.Module):
    def __init__(self, device=None, **fid_kwargs):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fid_tracker = FIDMetricsTracker(device=self.device, **fid_kwargs)
        self.image_tracker = ImageMetricsTracker(device=self.device)

    def reset(self):
        self.fid_tracker.reset()
        self.image_tracker.reset()

    @torch.no_grad()
    def update(self, target, pred):
        self.fid_tracker.update(target, pred)
        self.image_tracker.update(target, pred)

    @torch.no_grad()
    def aggregate(self):
        metrics_fid = self.fid_tracker.aggregate()
        metrics_img = self.image_tracker.aggregate()
        return {**metrics_fid, **metrics_img}




############################################
# Test functions
############################################
def show_images(tensor_batch, title=""):
    grid = make_grid(tensor_batch[:8].cpu(), nrow=4)
    np_img = grid.permute(1, 2, 0).numpy()
    plt.imshow(np_img)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
    
def plot_dist_hist(dist, title="Cross distances"):
    dists_flat = dist.cpu().flatten().numpy()
    plt.hist(dists_flat, bins=50, color='skyblue', alpha=0.7)
    plt.title(title)
    plt.xlabel("L2 distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

############################################
# Standalone PRDC Test (with NumPy)
############################################
def test_prdc_standalone():
    print("\n[Standalone PRDC NumPy Test]")
    from prdc import compute_prdc
    import numpy as np

    num_real_samples = num_fake_samples = 10000
    feature_dim = 1000
    nearest_k = 5

    real_features = np.random.normal(loc=0.0, scale=1.0, size=(num_real_samples, feature_dim))
    fake_features = np.random.normal(loc=0.0, scale=1.0, size=(num_fake_samples, feature_dim))

    prdc_metrics = compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=nearest_k
    )

    for key, val in prdc_metrics.items():
        print(f"{key:>16}: {val:.4f}")
    print("PRDC NumPy test passed.")


############################################
# ImageMetricsTracker Full Test
############################################
def test_fid_metrics_tracker():
    print("Running ImageMetricsTracker Test Suite...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = FIDMetricsTracker(num_crops=2, crop_size=64, device=device)
    tracker.reset()

    batch_size, C, H, W = 64, 3, 256, 256
    imgs_clean = torch.rand(batch_size, C, H, W, device=device)

    def run_case(name, target, pred, expect_low=False):
        tracker.reset()
        tracker.update(target, pred)
        results = tracker.aggregate()
        print(f"\n[{name.upper()}]")
        for k, v in results.items():
            print(f"{k:>18}: {v:.4f}")
        if expect_low:
            assert results["pFID"] < 1.0 and results["rFID"] < 1.0, "Expected low pFID/rFID"
        assert all(k in results for k in [
            "gfid", "pFID", "rFID",
            "prdc_precision", "prdc_recall", "prdc_density", "prdc_coverage",
        ]), "Missing keys in metrics output"

    # Identical images
    run_case("Identical", imgs_clean, imgs_clean.clone(), expect_low=True)

    # Noisy
    imgs_noisy = (imgs_clean + 0.2 * torch.randn_like(imgs_clean)).clamp(0, 1)
    run_case("Noisy", imgs_clean, imgs_noisy)

    # Random
    imgs_random = torch.rand(batch_size, 3, H, W, device=device)
    run_case("Random", imgs_clean, imgs_random)

    # Shifted
    imgs_shifted = TF.affine(imgs_clean, angle=0, translate=[10, 10], scale=1.0, shear=[0, 0])
    run_case("Shifted", imgs_clean, imgs_shifted)

    # Jittered
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    imgs_jittered = torch.stack([jitter(img.cpu()).to(device) for img in imgs_clean])
    run_case("Jittered", imgs_clean, imgs_jittered)

    print("\nAll ImageMetricsTracker tests passed successfully.")


def test_image_metrics_tracker():
    print("Running ImageMetricsTracker Test Suite...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = ImageMetricsTracker(device=device)
    tracker.reset()

    batch_size, C, H, W = 64, 3, 256, 256  # more reasonable for LPIPS/SSIM
    imgs_clean = torch.rand(batch_size, C, H, W, device=device)

    def run_case(name, target, pred, expect_low_lpips=False):
        tracker.reset()
        tracker.update(target, pred)
        results = tracker.aggregate()
        print(f"\n[{name.upper()}]")
        for k, v in results.items():
            print(f"{k:>10}: {v:.4f}")

        # Check required keys
        required_keys = ["ssim", "psnr", "mse", "mae", "lpips"]
        assert all(k in results for k in required_keys), "Missing metrics!"

        if expect_low_lpips:
            assert results["lpips"] < 0.1, f"Expected low LPIPS but got {results['lpips']:.4f}"

    # 1. Identical images (perfect metrics)
    run_case("Identical", imgs_clean, imgs_clean.clone(), expect_low_lpips=True)

    # 2. Additive Gaussian noise
    imgs_noisy = (imgs_clean + 0.2 * torch.randn_like(imgs_clean)).clamp(0, 1)
    run_case("Noisy", imgs_clean, imgs_noisy)

    # 3. Completely random images
    imgs_random = torch.rand(batch_size, C, H, W, device=device)
    run_case("Random", imgs_clean, imgs_random)

    # 4. Shifted images
    imgs_shifted = TF.affine(imgs_clean, angle=0, translate=[15, 15], scale=1.0, shear=[0, 0])
    run_case("Shifted", imgs_clean, imgs_shifted)

    # 5. Color jittered images
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    imgs_jittered = torch.stack([jitter(img.cpu()).to(device) for img in imgs_clean])
    run_case("Jittered", imgs_clean, imgs_jittered)

    print("\nAll ImageMetricsTracker tests passed successfully.")

    
    
############################################
# Main Execution
############################################
if __name__ == "__main__":
    seed_everything(2025)
    test_prdc_standalone()
    test_fid_metrics_tracker()
    test_image_metrics_tracker()
