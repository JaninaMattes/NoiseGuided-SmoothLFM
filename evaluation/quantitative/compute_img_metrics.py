# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
# - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/master/improved_precision_recall.py#L185
# - https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics

import os, sys
import csv
import gc
from tqdm import tqdm
import shutil
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from lightning import seed_everything
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from scipy import linalg

from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
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
        path.mkdir(parents=True, exist_ok=True)




##############################################
# Dataset to wrap loaded samples
##############################################
class SavedSamplesDataset(Dataset):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.fake_dir = self.base_dir / "fake-images"
        self.real_dir = self.base_dir / "gt-images"
        self.label_dir = self.base_dir / "labels"

        self.file_list = self._collect_valid_triplets()

        print(f"[INFO] Found {len(self.file_list)} complete samples in {self.base_dir}")
        assert self.file_list, f"No complete sample triplets found in {self.base_dir}"

    def _is_valid_id(self, name: str) -> bool:
        return name.isdigit()

    def _collect_valid_triplets(self):
        fake_ids = {f.stem for f in self.fake_dir.glob("*.png") if self._is_valid_id(f.stem)}
        real_ids = {f.stem for f in self.real_dir.glob("*.png") if self._is_valid_id(f.stem)}
        label_ids = {f.stem for f in self.label_dir.glob("*.npy") if self._is_valid_id(f.stem)}
        valid_ids = sorted(fake_ids & real_ids & label_ids)
        return [f"{fid}.png" for fid in valid_ids]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        fake_path = self.fake_dir / file_name
        real_path = self.real_dir / file_name
        label_path = self.label_dir / file_name.replace(".png", ".npy")

        try:
            fake_tensor = load_png_to_tensor_normalized(fake_path)
            real_tensor = load_png_to_tensor_normalized(real_path)
            label = np.load(label_path)
            label_tensor = torch.tensor(label.item() if not np.isscalar(label) else label, dtype=torch.long)
            return fake_tensor, real_tensor, label_tensor

        except Exception as e:
            print(f"[WARNING] Failed to load sample {file_name}: {e}")
            return None  # Must be handled via custom collate_fn



def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # drop empty batch

    collated = []
    for samples in zip(*batch):
        if isinstance(samples[0], torch.Tensor):
            collated.append(torch.stack(samples))
        else:
            collated.append(torch.tensor(samples))
    return tuple(collated)


def load_saved_samples_as_dataset(base_dir: str, batch_size: int = 8, shuffle: bool = False):
    dataset = SavedSamplesDataset(base_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True,
        collate_fn=safe_collate
    )



def save_tensor_as_png(tensor: torch.Tensor, path: str):
    """
    Save a torch tensor (C, H, W) as a .png image.
    Assumes tensor is in [-1, 1] or [0, 1].
    """
    tensor = denorm_tensor(tensor).detach().cpu() # just for visualization
    img = TF.to_pil_image(tensor.byte())
    img.save(path)


def load_png_to_tensor(path: str) -> torch.Tensor:
    """
    Load a .png image and convert to torch.Tensor in [0, 1], shape [3, H, W].
    """
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # Returns float tensor in [0, 255]


def load_png_to_tensor_normalized(path: str) -> torch.Tensor:
    """
    Load .png image and return tensor in [-1, 1], shape [3, H, W].
    """
    tensor = load_png_to_tensor(path)  # [0, 255]
    tensor = tensor / 127.5 - 1.0  # [0, 255] → [-1, 1]
    return tensor




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
#   FID class using InceptionV3
############################################
class PrecisionRecallFID(nn.Module):
    def __init__(self, num_k=5, device=None):
        super().__init__()

        try:
            num_k = int(num_k)
            if num_k < 1:
                raise ValueError("num_k must be ≥ 1")
        except (ValueError, TypeError):
            print(f"[WARNING] Invalid num_k={num_k}, defaulting to 5")
            num_k = 5

        self.num_k = num_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.device).eval()

        self.reset()

    def reset(self):
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
        # === Check for empty feature lists ===
        if not self.real_feats or not self.fake_feats:
            raise ValueError(
                f"Cannot compute PRDC — "
                f"{'real_feats is empty' if not self.real_feats else ''}"
                f"{' and ' if not self.real_feats and not self.fake_feats else ''}"
                f"{'fake_feats is empty' if not self.fake_feats else ''}."
            )

        real_feats = torch.cat(self.real_feats, dim=0).cpu().numpy()
        fake_feats = torch.cat(self.fake_feats, dim=0).cpu().numpy()

        # === Ensure k is valid for given sample sizes ===
        min_samples = min(len(real_feats), len(fake_feats))
        k_safe = min(self.num_k, min_samples - 1)
        
        if k_safe < 1:
            raise ValueError(f"Not enough samples to compute PRDC (need at least 2, got {min_samples})")

        return compute_prdc(
            real_features=real_feats,
            fake_features=fake_feats,
            nearest_k=k_safe
        )


    
    
    
############################################
# Image metrics tracker with pFID / rFID
############################################
class FIDMetricsTracker(nn.Module):
    def __init__(self, num_crops=4, crop_size=128, num_k=3, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_k = num_k

        self.global_fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

        self.prec_recall_fid = PrecisionRecallFID(num_k=self.num_k, device=self.device).to(self.device)

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
        self.prec_recall_fid.reset()
        if self.patch_fid:
            self.local_fid.reset()

    @torch.no_grad()
    def update(self, target, pred):
        real_ims = denorm_metrics_tensor(target, target_range=(0, 255), dtype='int').to(self.device)
        fake_ims = denorm_metrics_tensor(pred, target_range=(0, 255), dtype='int').to(self.device)

        if self.patch_fid:
            cropped_real, cropped_fake, anchors = [], [], []
            for i in range(real_ims.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                    real_ims[0], output_size=(self.crop_size, self.crop_size)))
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
def patchwise_cosine_similarity(pred, target, patch_size=32):
    B, C, H, W = pred.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Patch size must divide image dimensions"

    pred_patches = F.unfold(pred, kernel_size=patch_size, stride=patch_size)
    target_patches = F.unfold(target, kernel_size=patch_size, stride=patch_size)

    cosine = F.cosine_similarity(pred_patches, target_patches, dim=1)  # shape [B, num_patches]
    return cosine.mean()  # scalar


class ImageMetricsTracker(nn.Module):
    def __init__(self, device=None, patch_size=32):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size

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
        self.cosines = []               # global
        self.patchwise_cosines = []     # patch-wise

    @torch.no_grad()
    def update(self, target, pred):
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        pred_norm = denorm_metrics_tensor(pred, target_range=(0, 1), dtype='float').to(self.device)
        target_norm = denorm_metrics_tensor(target, target_range=(0, 1), dtype='float').to(self.device)

        self.ssims.append(self.ssim(pred_norm, target_norm).detach().cpu())
        self.psnrs.append(self.psnr(pred_norm, target_norm).detach().cpu())
        self.mses.append(torch.mean((pred_norm - target_norm) ** 2, dim=[1, 2, 3]).detach().cpu())
        self.maes.append(torch.mean(torch.abs(pred_norm - target_norm), dim=[1, 2, 3]).detach().cpu())
        self.lpips_scores.append(self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1).detach().cpu())

        # Global cosine similarity
        self.cosines.append(F.cosine_similarity(pred_norm.flatten(1), target_norm.flatten(1), dim=1).cpu())

        # Patch-wise cosine similarity
        patch_cossim = patchwise_cosine_similarity(pred_norm, target_norm, patch_size=self.patch_size)
        self.patchwise_cosines.append(patch_cossim.detach().cpu())

        del pred_norm, target_norm
        torch.cuda.empty_cache()


    @torch.no_grad()
    def aggregate(self):
        result = dict(
            ssim=torch.stack(self.ssims).mean().item() if self.ssims else 0.0,
            psnr=torch.stack(self.psnrs).mean().item() if self.psnrs else 0.0,
            mse=torch.stack(self.mses).mean().item() if self.mses else 0.0,
            mae=torch.stack(self.maes).mean().item() if self.maes else 0.0,
            lpips=torch.stack(self.lpips_scores).mean().item() if self.lpips_scores else 0.0,
            cosine=torch.stack(self.cosines).mean().item() if self.cosines else 0.0,
            patch_cossim=torch.stack(self.patchwise_cosines).mean().item() if self.patchwise_cosines else 0.0,
        )
        return result




#########################################################
#              Sample Generation                       #
#########################################################
@torch.no_grad()
def generate_samples(
    fm_module,
    images,
    xt_latent,
    labels=None,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    num_steps=50,
    num_classes=1000,
    nrow=4,
    title="Generated Samples",
    save_path=None,
    resize_to=128,
    use_labels=True,
    plot_samples=False,
    device=None,
    same_noise=True,
    z_global=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    xt_latent = xt_latent.to(device)
    
    if labels is not None:
        labels = labels.to(device).squeeze()
    else:
        lables = None # translated to Null class internally
        
    with torch.no_grad():
        context = fm_module.encode_third_stage(xt_latent)
        
        if same_noise:
            # Same noise for Ablation
            assert z_global is not None, "[ERROR] z_global is required when same_noise=True"
            z = z_global.clone()[:xt_latent.size(0)]

        else:
            # Different noise for real-world generation
            z = torch.randn_like(xt_latent, device=device)
            
            
        uc_context = torch.zeros_like(context).to(device)
        uc_label = torch.full((xt_latent.size(0),), num_classes, device=device, dtype=torch.long)

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
        real_images = images

    if plot_samples:
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
        del fig, grid, fake_images_, real_images_, real_images_resized, fake_images_resized, interleaved
        torch.cuda.empty_cache()

    del images, xt_latent, context, z, uc_context, uc_label, generated
    torch.cuda.empty_cache()

    return fake_images.detach().cpu(), real_images.detach().cpu()






##################################################
# Collect features
##################################################
@torch.no_grad()
def collect_real_and_fake_features(
    fm_module,
    dataloader,
    output_root="temp_storage",
    source_timestep=0.5,
    max_samples=25000,
    cfg_scale=3.0,
    ccfg_scale=1.0,
    num_steps=50,
    num_classes=1000,
    plot_samples=False,
    save_path=None,
    device=None,
    plot_every=1000,
    start_idx=0, 
    use_labels=True,
    same_noise_ablation=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(output_root)
    output_fake_dir = output_root / "fake-images"
    output_real_dir = output_root / "gt-images"
    output_label_dir = output_root / "labels"

    output_fake_dir.mkdir(parents=True, exist_ok=True)
    output_real_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing files to resume
    existing_files = sorted(output_fake_dir.glob("*.png"))
    if existing_files:
        last_file = existing_files[-1].stem
        assert last_file.isdigit(), f"Unexpected non-numeric filename: {last_file}"
        start_idx = int(last_file) + 1  # Start from next index
        count = len(existing_files)
        file_idx = start_idx
        print(f"[INFO] Resuming from existing samples. Found {count} samples, starting at index {file_idx}.")
    else:
        count = 0
        file_idx = start_idx
        print(f"[INFO] Starting fresh. No existing samples found, starting at index {file_idx}.")



    # Initialize the feature module
    for batch in tqdm(dataloader, desc="Collecting and saving images"):
        if count >= max_samples:
            print(f"[INFO] Already collected {count} samples, stopping early.")
            break

        images = batch["image"].to(device)
        xt_latent = batch[f"latents_{source_timestep:.2f}"].to(device)
        labels = batch["label"].to(device)

        # For comparability
        z_global = None
        if same_noise_ablation:
            z_global = torch.randn_like(xt_latent[:1], device=device).expand(xt_latent.size(0), -1, -1, -1).clone()

        fake_imgs, real_imgs = generate_samples(
            fm_module=fm_module,
            images=images,
            xt_latent=xt_latent,
            labels=labels,
            cfg_scale=cfg_scale,
            ccfg_scale=ccfg_scale,
            num_steps=num_steps,
            num_classes=num_classes,
            plot_samples=plot_samples if count % plot_every == 0 else False,
            save_path=save_path,
            device=device,
            use_labels=use_labels,
            same_noise=same_noise_ablation,
            z_global=z_global
        )

        batch_size = real_imgs.size(0)
        print(f"[INFO] Processing batch of size {batch_size} (count={count})")

        for i in range(batch_size):
            if count >= max_samples:
                print(f"[INFO] Reached max_samples {max_samples}.")
                break

            fake_img_tensor = fake_imgs[i]
            real_img_tensor = real_imgs[i]
            label = labels[i].cpu().numpy().astype(np.int64)

            filename = f"{file_idx:07d}.png"
            save_tensor_as_png(fake_img_tensor, output_fake_dir / filename)
            save_tensor_as_png(real_img_tensor, output_real_dir / filename)

            label_name = filename.replace(".png", ".npy")
            np.save(output_label_dir / label_name, label)


            file_idx += 1
            count += 1

        # Clean up memory
        del images, xt_latent, fake_imgs, real_imgs, labels

    print(f"All samples saved to: {output_root.resolve()}")




##################################################
# Get dataloader by group
##################################################
def get_dataloader_by_group(data_module, group: str):
    if group == "validation":
        print("[INFO] Using validation dataloader")
        return data_module.val_dataloader()
    elif group == "test":
        print("[INFO] Using test dataloader")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Unsupported group: {group}")




def get_last_valid_sample_index(fake_dir: Path, real_dir: Path, label_dir: Path) -> int:
    fake_ids = {int(f.stem) for f in fake_dir.glob("*.png") if f.stem.isdigit()}
    real_ids = {int(f.stem) for f in real_dir.glob("*.png") if f.stem.isdigit()}
    label_ids = {int(f.stem) for f in label_dir.glob("*.npy") if f.stem.isdigit()}

    common_ids = fake_ids & real_ids & label_ids
    if not common_ids:
        return 0                # No labels found

    return max(common_ids) + 1  # Resume from next index




##################################################
# Data collection and evaluation
##################################################
@torch.no_grad()
def run_data_collection_and_evaluation(
    checkpoint,
    data_path,
    project_name,
    model_name,
    group="validation",
    source_timestep=0.50,
    target_timestep=1.00,
    cfg_scales=[1.0, 2.0, 3.0],
    ccfg_scales=[1.0, 1.0, 1.0],
    class_conditional=True,
    batch_size=32,
    max_samples=50000,
    max_samples_to_compare=1000,
    num_steps=50,
    num_classes=1000,
    results_root="results",
    num_crops=4,
    crop_size=128,
    num_k=3,
    device=None,
    sample_output_root=None,
    force_clean=False,
    same_noise_ablation=False
):
    assert len(cfg_scales) == len(ccfg_scales), "Mismatch: cfg_scales and ccfg_scales must be same length"
    if not exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # Set device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm_module = TrainerModuleLatentFlow.load_from_checkpoint(checkpoint, map_location="cpu")
    fm_module.eval().to(device)

    # Load the data module
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
    dataloader = get_dataloader_by_group(data, group)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)

    sample_dir_base = Path(sample_output_root) if sample_output_root else None
    if sample_dir_base:
        sample_dir_base.mkdir(parents=True, exist_ok=True)

    class_cond_tag = 'ccfg' if class_conditional else 'cfg'
    csv_path = base_results_dir / f"{model_name}_{class_cond_tag}_metrics.csv"
    csv_header = [
        "cfg_scale", "ccfg_scale",
        "gFID", "lFID",
        "PRDC_Precision", "PRDC_Recall", "PRDC_Density", "PRDC_Coverage",
        "TF_FID", "TF_KID_Mean", "TF_KID_Std",
        "TF_ISC_Mean", "TF_ISC_Std", "TF_Precision", "TF_Recall",
        "SSIM", "LPIPS", "PSNR", "MSE", "MAE", "CosSim", "Patch_CosSim",
        "Chunks_Aggregated", "Samples_Processed"
    ]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    for cfg_scale, ccfg_scale in zip(cfg_scales, ccfg_scales):
        print(f"\n[INFO] CFG={cfg_scale}, CCFG={ccfg_scale}")
        sample_dir = (sample_dir_base or base_results_dir / f"samples_cfg{cfg_scale}_ccfg{ccfg_scale}")
        fake_dir, real_dir, label_dir = sample_dir / "fake-images", sample_dir / "gt-images", sample_dir / "labels"
        for folder in [fake_dir, real_dir, label_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        if force_clean:
            clear_folder(fake_dir)
            clear_folder(real_dir)
            clear_folder(label_dir)
            start_idx, skip_sampling = 0, False
        else:
            start_idx = get_last_valid_sample_index(fake_dir, real_dir, label_dir)
            skip_sampling = start_idx >= max_samples

        if not skip_sampling:
            collect_real_and_fake_features(
                fm_module=fm_module,
                dataloader=dataloader,
                output_root=sample_dir,
                source_timestep=source_timestep,
                max_samples=max_samples,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                num_steps=num_steps,
                num_classes=num_classes,
                device=device,
                start_idx=start_idx,
                use_labels=class_conditional,
                plot_samples=(sample_dir_base is not None and sample_dir_base.exists() and start_idx == 0),
                same_noise_ablation=same_noise_ablation
            )
            
            
        # ----------------------------------------------
        # Evaluation
        # ----------------------------------------------
        sample_loader = load_saved_samples_as_dataset(
            sample_dir,
            batch_size=batch_size,
            shuffle=False
        )

        fid_tracker = FIDMetricsTracker(num_crops=num_crops, crop_size=crop_size, num_k=num_k, device=device)
        torch_fidelity_tracker = TorchFidelityFIDTracker(fake_path=fake_dir, gt_path=real_dir, device=device)
        img_tracker = ImageMetricsTracker(device=device)

        accum_metrics = {k: 0.0 for k in [
            'gfid', 'lfid', 'prdc_precision', 'prdc_recall', 'prdc_density', 'prdc_coverage',
            'tf_fid', 'tf_kid_mean', 'tf_kid_std', 'tf_isc_mean', 'tf_isc_std',
            'tf_precision', 'tf_recall', 'ssim', 'lpips', 'psnr', 'mse', 'mae', 'cosine', 'patch_cossim'
        ]}

        lfid_batches, total_sample_count, chunk_sample_count, chunk_counter = 0, 0, 0, 0
        print(f"[INFO] Starting evaluation for CFG={cfg_scale}, CCFG={ccfg_scale}")
        print(f"[INFO] Max samples to compare: {max_samples_to_compare}, Max samples: {max_samples}")

        # Initialize trackers
        fid_tracker.reset()
        img_tracker.reset()
        
        for batch in tqdm(sample_loader, desc=f"Evaluating CFG={cfg_scale}"):
            if batch is None:
                print("[WARNING] Skipping empty batch")
                continue

            # Leave sampler
            if total_sample_count >= max_samples:
                print(f"[INFO] Reached max_samples {max_samples}. Stopping evaluation.")
                break

            try:
                # Collect samples
                batch_fake, batch_real, _ = batch
                batch_fake, batch_real = batch_fake.to(device), batch_real.to(device)

                fid_tracker.update(batch_real, batch_fake)
                img_tracker.update(batch_real, batch_fake)
                total_sample_count += batch_fake.size(0) # Track total samples processed
                chunk_sample_count += batch_fake.size(0)
                
            except Exception as e:
                print(f"[WARNING] Skipping corrupt batch: {e}")
            
            try:
                # Compute metrics
                if chunk_sample_count >= max_samples_to_compare:
                    # compute metrics for this chunk
                    fid_metrics = fid_tracker.aggregate()
                    img_metrics = img_tracker.aggregate()

                    combined_metrics = {**fid_metrics, **img_metrics}
                    for k in accum_metrics:
                        if k in combined_metrics and combined_metrics[k] is not None:
                            value = combined_metrics.get(k, 0.0)
                            accum_metrics[k] += value
                            if k == 'lfid':
                                lfid_batches += 1
                    
                    # Memorise the number of chunks processed
                    chunk_sample_count = 0
                    chunk_counter += 1
                    
                    # Reset trackers for next chunk
                    fid_tracker.reset(), img_tracker.reset()

                    print("=" * 50)
                    print(f"[INFO] Aggregated metrics after {chunk_counter} chunks.")
                    print(f"    gFID={fid_metrics.get('gfid', float('nan')):.6f}, lFID={fid_metrics.get('lfid', float('nan')):.6f}")
                    print(f"    PRDC Precision={fid_metrics.get('prdc_precision', float('nan')):.6f}, Recall={fid_metrics.get('prdc_recall', float('nan')):.6f}")
                    print(f"    SSIM={img_metrics.get('ssim', float('nan')):.6f}, LPIPS={img_metrics.get('lpips', float('nan')):.6f}, PSNR={img_metrics.get('psnr', float('nan')):.2f}")
                    print(f"    Cosine={img_metrics.get('cosine', float('nan')):.6f}, Patch CosSim={img_metrics.get('patch_cossim', float('nan')):.6f}")
                    print("=" * 50)
                
            except Exception as e:
                print(f"[ERROR] Failed to compute metrics for batch: {e}")
        
        try:
            if chunk_sample_count > 0:
                fid_metrics = fid_tracker.aggregate()
                img_metrics = img_tracker.aggregate()
                combined_metrics = {**fid_metrics, **img_metrics}

                for k in accum_metrics:
                    if k in combined_metrics and combined_metrics[k] is not None:
                        accum_metrics[k] += combined_metrics[k]
                        if k == 'lfid':
                            lfid_batches += 1
                chunk_counter += 1
                print(f"[INFO] Final aggregation after last chunk: {chunk_counter} chunks processed.")
            else:
                print("[INFO] No samples processed, skipping final aggregation.")
        except Exception as e:
            print(f"[ERROR] Failed to aggregate final metrics: {e}")
        finally:
            # Reset trackers for next chunk
            fid_metrics = fid_tracker.aggregate() if chunk_counter > 0 else {}
            img_metrics = img_tracker.aggregate() if chunk_counter > 0 else {}
            fid_tracker.reset()
            img_tracker.reset()

        # Compute torch-fidelity metrics
        try:
            tf_metrics = torch_fidelity_tracker.compute()
        except Exception as e:
            print(f"[ERROR] torch_fidelity: {e}")
            tf_metrics = {k: float('nan') for k in [
                "tf_fid", "tf_kid_mean", "tf_kid_std", "tf_isc_mean",
                "tf_isc_std", "tf_precision", "tf_recall"
            ]}

        denom = chunk_counter or 1
        fid_avg = {
            'gfid': accum_metrics['gfid'] / denom,
            'lfid': accum_metrics['lfid'] / lfid_batches if lfid_batches else None,
            'prdc_precision': accum_metrics['prdc_precision'] / denom,
            'prdc_recall': accum_metrics['prdc_recall'] / denom,
            'prdc_density': accum_metrics['prdc_density'] / denom,
            'prdc_coverage': accum_metrics['prdc_coverage'] / denom,
        }
        img_avg = {k: accum_metrics[k] / denom for k in ['ssim', 'lpips', 'psnr', 'mse', 'mae', 'cosine', 'patch_cossim']}

        def fval(val, fmt): return fmt.format(val) if val is not None else "NA"
        
        
        # Prepare CSV row
        csv_row = [
            cfg_scale, ccfg_scale,
            fval(fid_avg['gfid'], '{:.6f}'),
            fval(fid_avg['lfid'], '{:.6f}'),
            fval(fid_avg['prdc_precision'], '{:.6f}'),
            fval(fid_avg['prdc_recall'], '{:.6f}'),
            fval(fid_avg['prdc_density'], '{:.6f}'),
            fval(fid_avg['prdc_coverage'], '{:.6f}'),
            fval(tf_metrics['tf_fid'], '{:.6f}'),
            fval(tf_metrics['tf_kid_mean'], '{:.6f}'),
            fval(tf_metrics['tf_kid_std'], '{:.6f}'),
            fval(tf_metrics['tf_isc_mean'], '{:.6f}'),
            fval(tf_metrics['tf_isc_std'], '{:.6f}'),
            fval(tf_metrics['tf_precision'], '{:.6f}'),
            fval(tf_metrics['tf_recall'], '{:.6f}'),
            fval(img_avg['ssim'], '{:.6f}'),
            fval(img_avg['lpips'], '{:.6f}'),
            fval(img_avg['psnr'], '{:.2f}'),
            fval(img_avg['mse'], '{:.6f}'),
            fval(img_avg['mae'], '{:.6f}'),
            fval(img_avg['cosine'], '{:.6f}'),
            fval(img_avg['patch_cossim'], '{:.6f}'),
            chunk_counter,      # How many chunks were processed
            total_sample_count  # How many samples were processed in total
        ]
        
        print("[DEBUG] Writing row to CSV:", csv_row)
        print("[DEBUG] CSV path:", csv_path.resolve())
        
        
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(csv_row)
            print(f"[INFO] Appended metrics to {csv_path.resolve()}")

        
        print(f"=" * 50)
        print(f"[INFO] Metrics for CFG={cfg_scale}, CCFG={ccfg_scale}:"
              f"\n    gFID={fid_avg['gfid']:.6f}, lFID={fid_avg['lfid']:.6f}"
              f"\n    PRDC Precision={fid_avg['prdc_precision']:.6f}, Recall={fid_avg['prdc_recall']:.6f}"
              f"\n    SSIM={img_avg['ssim']:.6f}, LPIPS={img_avg['lpips']:.6f}, PSNR={img_avg['psnr']:.2f}"
              f"\n    Cosine={img_avg['cosine']:.6f}, Patch CosSim={img_avg['patch_cossim']:.6f}"
              f"\n    TF FID={tf_metrics['tf_fid']:.6f}, KID Mean={tf_metrics['tf_kid_mean']:.6f}, KID Std={tf_metrics['tf_kid_std']:.6f}"
              f"\n    ISC Mean={tf_metrics['tf_isc_mean']:.6f}, ISC Std={tf_metrics['tf_isc_std']:.6f}"
              f"\n    TF Precision={tf_metrics['tf_precision']:.6f}, TF Recall={tf_metrics['tf_recall']:.6f}")
        print(f"[INFO] Saved metrics to {csv_path.resolve()}")
        print(f"=" * 50)
        
        # Clean up temporary folders
        clear_folder(fake_dir)
        clear_folder(real_dir)
        clear_folder(label_dir)
        print(f"[INFO] Finished evaluation for CFG={cfg_scale}, cleaned: {sample_dir}")




if __name__ == "__main__":

    #####################################
    # Shared temp folder
    #####################################
    test_data_path      = './dataset/processed/testset-256/imagenet256-testset-T151412.hdf5'
    val_data_path       = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5'
    temp_folder         = './results/temp_storage/SiT-XL-2/test/'
    results_root        = './results/CFG_Eval_Class-Conditional/'
    class_conditional   = True  # Set to True for standard class-conditional


    #####################################
    # Evaluation Parameters
    #####################################
    source_timestep     = 0.50
    target_timestep     = 1.00
    beta                = 0.1
    dataset_name        = 'imagenet256-dataset-T000006'
    group               = "validation"  # "validation" or "test"
    batch_size          = 32
    samples_per_class   = 14
    num_pairs           = 12
    num_interpolations  = 20
    cfg_scale           = 4.0
    ccfg_scale          = 1.0


    #####################################
    # Model Paths
    #####################################
    # beta: 0.1
    DiTSXL_Beta00x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA05x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt'  
    DITSXL_BETA02x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-08/30859/checkpoints/last.ckpt'   ### Done
    DiTSXL_Beta05x05x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt'   ### (Baseline)
    
    # beta: 0.5
    DITSXL_Beta02x10x_05b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_0.5b/BetaVAE-B-2/V0/2025-07-20/31735/checkpoints/last.ckpt'   ### Done
    
    # beta: 1.0
    DITSXL_Beta05x05x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt' 
    DITSXL_Beta05x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt'   
    DITSXL_Beta02x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt'   

    
    # beta: 2.0
    DiTSXL_Beta02x10x_2b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_2.0b/BetaVAE-B-2/V0/2025-07-17/31522/checkpoints/last.ckpt'

    # beta: 5.0
    DiTSXL_Beta05x05x_5b ='./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt'
    DiTSXL_Beta02x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt'    
    DiTSXL_Beta05x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt' 





    #####################################
    # Model Configurations
    #####################################
    model_configs = [
        # ---------------------------
        # beta: 0.1
        # ---------------------------
        {
            "checkpoint": DITSXL_BETA05x10x_01b,
            "source_timestep": 0.50,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1" 
              #TODO:  CFG: 5, 7, 9 missing
        },
        {
            "checkpoint": DITSXL_BETA02x10x_01b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.1,
            "group_name": "Beta0.1"
            #TODO:  CFG: 5, 7, 9 missing
        },
        # {
        #     "checkpoint": DiTSXL_Beta00x10x_01b,
        #     "source_timestep": 0.00,
        #     "target_timestep": 1.00,
        #     "beta": 0.1,
        #     "group_name": "Beta0.1"
        # },        
        # {
        #     "checkpoint": DiTSXL_Beta05x05x_01b,
        #     "source_timestep": 0.50,
        #     "target_timestep": 0.50,
        #     "beta": 0.1,
        #     "group_name": "Beta0.1_Baseline"
        # },
        # ---------------------------
        # beta: 0.5
        # ---------------------------
        {   
            "checkpoint": DITSXL_Beta02x10x_05b,
            "source_timestep": 0.20,
            "target_timestep": 1.00,
            "beta": 0.5,
            "group_name": "Beta0.5",
            #TODO:  CFG: 5, 7, 9 missing
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
            #TODO:  CFG: 5, 7, 9 missing
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
             # TODO:  CFG: 5, 7, 9 missing
        },
        # # ---------------------------
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
            #TODO:  CFG: 5, 7, 9 missing
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

    #####################################
    # Evaluation Parameters
    #####################################
    max_samples             = 20000                                 # Maximum number of samples to collect (50k is standard practice for FID)
    max_samples_to_compare  = 10000                                 # Maximum number of samples to compare for FID/LPIPS (50k is standard practice)
    batch_size              = 256                                   # Batch size for evaluation (smaller, e.g., 16 or 32)
    num_steps               = 50                                    # Fixed number of reverse-time steps for generation
    num_classes             = 1000                                  # Number of classes in the dataset (e.g., 1000 for ImageNet)
    cfg_scales              = [1.0, 3.0, 5.0, 7.0, 9.0]             # [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]        # Class-conditional CFG scales (1.0 for no class conditioning)
    ccfg_scales             = [1.0, 1.0, 1.0, 1.0, 1.0]             # Class-conditional CFG scales (1.0 for no class conditioning)
    num_crops               = 4                                      # Number of crops for FID evaluation
    crop_size               = 128                                    # Crop size for FID evaluation
    num_k                   = 5                                      # Number of k neighbors for Torch-Fidelity evaluation
    same_noise_ablation     = True                                  # Whether to use the same noise for all samples (for ablation study all noises are kept constant)

    #####################################
    # Run unified collection and evaluation
    #####################################
    for config in model_configs:
        checkpoint = config["checkpoint"]
        source_timestep = config["source_timestep"]
        target_timestep = config["target_timestep"]
        beta = config["beta"]
        group_name = config["group_name"]

        baseline = (source_timestep == 0.50 and target_timestep == 0.50)

        # Construct model name and project name
        model_name = f"{group_name}_VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"
        class_conditional_label = "Cls_Cond" if class_conditional else "No_Cls_Cond"
        model_name = f"{model_name}_{class_conditional_label}"
        project_name = f"CFG_Quantitative_Eval_Baseline" if baseline else "CFG_Quantitative_Eval"
        
        print(f"\n{'='*100}\nEvaluating model: {model_name}\n{'='*100}\n")
        print(f"Class Conditional: {class_conditional}")

        # CLEAR THE OUTPUT FOLDER
        output_dir = Path(temp_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*100}\nStarting evaluation for: {model_name}\n{'='*100}\n")
        run_data_collection_and_evaluation(
            checkpoint=checkpoint,
            data_path=test_data_path if group == "test" else val_data_path,
            project_name=project_name,
            model_name=model_name,
            group=group,
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            cfg_scales=cfg_scales,
            ccfg_scales=ccfg_scales,
            class_conditional=class_conditional, # With label or without
            batch_size=batch_size,
            max_samples=max_samples,
            max_samples_to_compare=max_samples_to_compare,
            num_steps=num_steps,
            num_classes=num_classes,
            results_root=results_root,
            num_crops=num_crops,
            crop_size= crop_size,
            num_k=num_k,
            sample_output_root=output_dir,  # Pass cleaned path here
            same_noise_ablation=same_noise_ablation
        )
        print(f"\n{'='*100}\nFinished evaluation for: {model_name}\n{'='*100}\n")
        torch.cuda.empty_cache()
        gc.collect()

    

# CUDA_VISIBLE_DEVICES=0 python ...