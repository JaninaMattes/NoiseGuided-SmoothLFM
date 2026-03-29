
# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import shutil
import os, sys
import gc

from tqdm import tqdm

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

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


import torch_fidelity
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

from ldm.trainer import TrainerModuleLatentFlow
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
        # convert to float32 and normalize to [0, 1]
        images = denorm_metrics_tensor(images, target_range=(0, 1), dtype='float').to(self.device)
        if images.dtype != torch.float32:
            raise ValueError(f"Expected images to be float32, got {images.dtype}")
        
        feats = self.inception(images.to(self.device))[0].squeeze(-1).squeeze(-1)
        if real:
            self.real_feats.append(feats)
        else:
            self.fake_feats.append(feats)

    @torch.no_grad()
    def compute_prdc_metrics(self):
        real_feats = torch.cat(self.real_feats, dim=0).cpu().numpy()
        fake_feats = torch.cat(self.fake_feats, dim=0).cpu().numpy()

        return compute_prdc(
            real_features=real_feats,
            fake_features=fake_feats,
            nearest_k=self.k
        )

    
    
    
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

        self.prec_recall_fid = PrecisionRecallFID(k=k, device=self.device).to(self.device)

        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[FIDMetricsTracker] Using patch-wise FID")
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
        self.prec_recall_fid.real_feats = []
        self.prec_recall_fid.fake_feats = []
        if self.patch_fid:
            self.local_fid.reset()

    @torch.no_grad()
    def update(self, target, pred):
        real_ims = denorm_metrics_tensor(target, target_range=(0, 255), dtype='int').to(self.device)
        fake_ims = denorm_metrics_tensor(pred, target_range=(0, 255), dtype='int').to(self.device)

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

        self.prec_recall_fid.update(real_ims, real=True)
        self.prec_recall_fid.update(fake_ims, real=False)

        del real_ims, fake_ims
        torch.cuda.empty_cache()

    @torch.no_grad()
    def aggregate(self):
        prdc = self.prec_recall_fid.compute_prdc_metrics()
        gfid = self.global_fid.compute().item()
        lfid = self.local_fid.compute().item() if self.patch_fid else None
        
        return dict(
            gfid=max(gfid, 0.0),
            lfid=max(lfid, 0.0) if lfid is not None else None,
            prdc_precision=prdc['precision'],
            prdc_recall=prdc['recall'],
            prdc_density=prdc['density'],
            prdc_coverage=prdc['coverage'],
        )



############################################
# Torch-Fidelity FID tracker
############################################
class TorchFidelityFIDTracker(nn.Module):
    def __init__(self, fake_path: str, gt_path: str, num_images: int = 10, device=None):
        super().__init__()
        self.fake_path = str(fake_path)
        self.gt_path = str(gt_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def compute(self):
        try:
            # Get number of files in path 
            num_fake = len(os.listdir(self.fake_path))
            num_gt = len(os.listdir(self.gt_path))
            if num_fake < 10 or num_gt < 10:
                raise ValueError(f"Not enough images in fake ({num_fake}) or gt ({num_gt}) directories. Need at least 10 each.")
            
            print(f"[torch_fidelity] Computing metrics for {num_fake} fake images and {num_gt} ground truth images...")
            # Use torch-fidelity to compute metrics
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=self.fake_path,
                input2=self.gt_path,
                cuda=self.device.type == "cuda",
                isc=True, 
                fid=True, 
                kid=True,
                prc=True,
                verbose=False,
                kid_subset_size=min(64, num_fake, num_gt)  # Use a subset size of 64 or less if fewer images are available
            )


        except Exception as e:
            print(f"[ERROR] Torch-Fidelity failed: {e}")
            return {
                "tf_fid": float('nan'),
                "tf_kid_mean": float('nan'),
                "tf_kid_std": float('nan'),
                "tf_isc_mean": float('nan'),
                "tf_isc_std": float('nan'),
                "tf_precision": float('nan'),
                "tf_recall": float('nan'),
            }

        print(f"\n[torch_fidelity] Results: {metrics_dict}")

        return {
            "tf_fid": metrics_dict.get("frechet_inception_distance", float("nan")),
            "tf_kid_mean": metrics_dict.get("kernel_inception_distance_mean", float("nan")),
            "tf_kid_std": metrics_dict.get("kernel_inception_distance_std", float("nan")),
            "tf_isc_mean": metrics_dict.get("inception_score_mean", float("nan")),
            "tf_isc_std": metrics_dict.get("inception_score_std", float("nan")),
            "tf_precision": metrics_dict.get("precision", float("nan")),
            "tf_recall": metrics_dict.get("recall", float("nan")),
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
        )


############################################
#   Utils
############################################
from filelock import FileLock

def clear_folder(folder_path: Path):
    lock_file = folder_path.with_suffix('.lock')
    with FileLock(lock_file):
        try:
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"[INFO] Removed existing folder: {folder_path}")
            elif folder_path.exists() and folder_path.is_file():
                folder_path.unlink()
                print(f"[INFO] Removed existing file: {folder_path}")
            else:
                print(f"[INFO] Folder {folder_path} does not exist, nothing to clear.")
        except Exception as e:
            print(f"[ERROR] Failed to clear folder {folder_path}: {e}")
            raise RuntimeError(f"[ERROR] Failed to clear folder {folder_path}: {e}")
        finally:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created folder: {folder_path}")

############################################
# Test functions
############################################

def save_image(sample, filename, denorm=False):
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    if denorm:
        sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]
    sample = sample.clamp(0, 1)
    torchvision.utils.save_image(sample, filename, normalize=False)


def create_dummy_folder(path: str, num_images: int = 16, image_size=(3, 128, 128)):
    os.makedirs(path, exist_ok=True)
    for i in range(num_images):
        img = torch.rand(*image_size) * 2 - 1  # [-1, 1] range
        save_image(img, os.path.join(path, f"img_{i:03d}.png"), denorm=True)
    print(f"Created dummy folder at {path} with {num_images} images of size {image_size}")
    


def test_full_image_metrics_tracker():
    print("\n=== Running full ImageMetricsTracker end-to-end tests ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = ImageMetricsTracker(device=device)

    imgs = torch.rand(64, 3, 256, 256, device=device) * 2 - 1  # In range [-1, 1]
    cases = {
        "identical": (imgs, imgs.clone()),
        "noisy": (imgs, (imgs + 0.1 * torch.randn_like(imgs)).clamp(-1, 1)),
        "random": (imgs, torch.rand_like(imgs) * 2 - 1)
    }

    for name, (real, fake) in cases.items():
        tracker.reset()
        tracker.update(real, fake)
        metrics = tracker.aggregate()
        print(f"\n-- Case: {name} metrics --")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")


def test_fid_metrics_tracker():
    print("\n=== Running FIDMetricsTracker end-to-end tests ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = FIDMetricsTracker(device=device)

    imgs = torch.rand(64, 3, 256, 256, device=device) * 2 - 1  # In range [-1, 1]
    cases = {
        "identical": (imgs, imgs.clone()),
        "noisy": (imgs, (imgs + 0.1 * torch.randn_like(imgs)).clamp(-1, 1)),
        "random": (imgs, torch.rand_like(imgs) * 2 - 1)
    }

    for name, (real, fake) in cases.items():
        tracker.reset()
        tracker.update(real, fake)
        metrics = tracker.aggregate()
        print(f"\n-- Case: {name} metrics --")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
            
            
            
def test_torchfid_metrics_tracker(num_samples=100):
    print("\n=== Running TorchFidelityFIDTracker test ===")
    fake_path = "tmp/fake"
    real_path = "tmp/real"
    create_dummy_folder(fake_path, num_images=num_samples)
    create_dummy_folder(real_path, num_images=num_samples)

    tracker = TorchFidelityFIDTracker(fake_path, real_path)
    metrics = tracker.compute()

    print("\n-- TorchFidelity metrics --")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    clear_folder(Path(fake_path))
    clear_folder(Path(real_path))


def test_torch_fidelity_metrics_large_set(fake_dir="tmp/fake", real_dir="tmp/real", num_samples=5000):
    """
    Tests torch-fidelity with enough images to compute precision and recall.
    Creates synthetic fake/real samples.
    """
    print("\n=== Running TorchFidelityFIDTracker test ===")
    fake_path = "tmp/fake"
    real_path = "tmp/real"
    create_dummy_folder(fake_path, num_images=num_samples)
    create_dummy_folder(real_path, num_images=num_samples)

    tracker = TorchFidelityFIDTracker(fake_path, real_path, num_images=num_samples)
    metrics = tracker.compute()

    print("\n-- TorchFidelity metrics --")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    clear_folder(Path(fake_path))
    clear_folder(Path(real_path))
    
    # Check precision/recall
    assert "precision" in metrics and "recall" in metrics, "Precision/Recall not computed! Increase image count."
    assert metrics["precision"] > 0.0 and metrics["recall"] > 0.0, "Precision/Recall should be greater than 0!"






    
if __name__ == "__main__":
    
    try:
        # Create temporary directory for testing
        os.makedirs("tmp", exist_ok=True)
        test_full_image_metrics_tracker()
        test_fid_metrics_tracker()
        test_torchfid_metrics_tracker()
        test_torch_fidelity_metrics_large_set()
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
    finally:
        # Clean up temporary directory
        if os.path.exists("tmp"):
            clear_folder(Path("tmp"))
            print("[INFO] Cleaned up temporary directory.")
        else:
            print("[INFO] No temporary directory to clean up.")