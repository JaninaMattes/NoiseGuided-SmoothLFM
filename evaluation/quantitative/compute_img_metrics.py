from pytorch_fid.inception import InceptionV3

# Code adapted from:
# - https://github.com/SHI-Labs/Smooth-Diffusion
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
from scipy import linalg

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import numpy as np
from typing import Tuple
import pandas as pd

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

        # Clean up memory
        del real_feats, fake_feats, dists_real, dists_fake, dists_cross, precision_mask, recall_mask
        torch.cuda.empty_cache()
        gc.collect()
        
        return pFID, rFID

    def _pairwise(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp(min=0).sqrt()


    @torch.no_grad()
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
        
        # Cleam up memory
        del mu1, mu2, sigma1, sigma2, diff, cov_prod, covmean
        torch.cuda.empty_cache()
        gc.collect()
        
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

        # Clean up memory
        del real_feats, fake_feats, dists_real, dists_fake, dists_cross, precision_mask, recall_mask
        torch.cuda.empty_cache()
        gc.collect()
        
        return precision, recall

    @torch.no_grad()
    def _pairwise_distances(self, x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        dist = x_norm + y_norm - 2.0 * x @ y.t()
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
                    cropped_real.append(FT.crop(img_real, *anchor))
                    cropped_fake.append(FT.crop(img_fake, *anchor))
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
        gc.collect()

    @torch.no_grad()
    def aggregate(self):
        precision_val, recall_val = self.prec_recall.compute()
        pFID_val, rFID_val = self.prec_recall_fid.compute_pFID_rFID()

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
        self.lpips_scores.append(self.lpips(pred_norm * 2 - 1, target_norm * 2 - 1).detach().cpu())
        
        # Clean up memory
        del pred_norm, target_norm
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def aggregate(self):
        return dict(
            ssim=torch.stack(self.ssims).mean().item(),
            psnr=torch.stack(self.psnrs).mean().item(),
            mse=torch.stack(self.mses).mean().item(),
            mae=torch.stack(self.maes).mean().item(),
            lpips=torch.stack(self.lpips_scores).mean().item(),
        )




##############################################
# Dataset to wrap loaded samples
##############################################
class SavedSamplesDataset(Dataset):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.fake_dir = self.base_dir / "fake-images"
        self.real_dir = self.base_dir / "gt-images"
        self.label_dir = self.base_dir / "labels"

        self.file_list = sorted(self.fake_dir.glob("*.npy"))
        assert len(self.file_list) > 0, f"No .npy files found in {self.fake_dir}"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx].name

        fake_img = np.load(self.fake_dir / file_name)
        real_img = np.load(self.real_dir / file_name)
        label = np.load(self.label_dir / file_name).item()

        fake_tensor = torch.from_numpy(fake_img).float()
        real_tensor = torch.from_numpy(real_img).float()
        label_int = int(label)

        return fake_tensor, real_tensor, label_int

##############################################
# Load samples as dataset
##############################################
def load_saved_samples_as_dataset(base_dir: str, batch_size: int = 8, shuffle: bool = False):
    dataset = SavedSamplesDataset(base_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader




#########################################################
#              Sample Generation                       #
#########################################################
torch.no_grad()
def generate_samples(
    fm_module,
    images,
    xt_latent,
    labels=None,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    num_steps=50,
    num_classes=1000,
    device=None,
    denorm_fn=None,  # If needed later
    plot_samples=False,
    nrow=4,
    title="Generated Samples",
    save_path=None,
    resize_to=128,
    use_labels=False,  # If labels are used in the model
):
    device = device or fm_module.device

    # Move tensors to device
    images = images.to(device)
    xt_latent = xt_latent.to(device)
    
    if labels is not None:
        labels = labels.to(device).squeeze()

    with torch.no_grad():
        context = fm_module.encode_third_stage(xt_latent)
        z = torch.randn_like(xt_latent).to(device)          # noise tensor

        uc_context = torch.zeros_like(context).to(device)   # unconditional context
        uc_label = torch.full((xt_latent.size(0),), num_classes, device=device, dtype=torch.long)

        sample_kwargs = {
            "num_steps": num_steps,
            "progress": False,
            "context": context,
            "y": labels if use_labels else None,
            "cfg_scale": cfg_scale,
            "ccfg_scale": ccfg_scale,
            "uc_cond_context": uc_context,
            "uc_cond": uc_label,
        }

        generated = fm_module.model.generate(x=z, **sample_kwargs)
        fake_images = fm_module.decode_first_stage(generated)
        real_images = images  # Already unnormalized

    ##############################
    # Plotting (optional)
    ##############################
    # Warning: Denorm is okay for visualization, but not for metrics (!)
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
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    ############################
    # Clean up
    ############################
    del images, xt_latent, context, z, uc_context, uc_label, generated
    torch.cuda.empty_cache()
    gc.collect()

    return fake_images.detach().cpu(), real_images.detach().cpu()





##################################################
# Collect features
##################################################
@torch.no_grad()
def collect_real_and_fake_features(
    fm_module,
    data_loader,
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
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(output_root)
    output_fake_dir = output_root / "fake-images"
    output_real_dir = output_root / "gt-images"
    output_label_dir = output_root / "labels"

    # Create folders
    output_fake_dir.mkdir(parents=True, exist_ok=True)
    output_real_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    file_idx = 0

    for batch in tqdm(data_loader, desc="Collecting and saving images"):
        if count >= max_samples:
            print(f"[INFO] Collected {count} samples, stopping early.")
            break

        images = batch["image"].to(device)
        xt = batch[f"latents_{source_timestep:.2f}"].to(device)
        labels = batch["label"].to(device)

        fake_imgs, real_imgs = generate_samples(
            fm_module=fm_module,
            images=images,
            xt_latent=xt,
            labels=labels,
            cfg_scale=cfg_scale,
            ccfg_scale=ccfg_scale,
            num_steps=num_steps,
            num_classes=num_classes,
            plot_samples=plot_samples if count % plot_every == 0 else False,
            save_path=save_path,
            device=device,
        )

        batch_size = real_imgs.size(0)

        # Iterate and save one by one
        for i in range(batch_size):
            fake_img = fake_imgs[i].cpu().numpy().astype(np.float32)  # Shape (C, H, W)
            real_img = real_imgs[i].cpu().numpy().astype(np.float32)
            label = labels[i].cpu().numpy().astype(np.int64)  # single value

            # Format filename
            filename = f"{file_idx:07d}.npy"

            # Save
            np.save(output_fake_dir / filename, fake_img)
            np.save(output_real_dir / filename, real_img)
            np.save(output_label_dir / filename, label)

            file_idx += 1
            count += 1

            if count >= max_samples:
                print(f"[INFO] Reached max_samples {max_samples}.")
                break

        # Clean up
        del images, xt, labels, fake_imgs, real_imgs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[✓] All samples saved to: {output_root.resolve()}")





def load_saved_samples(base_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Load all saved samples from `base_dir`.

    Returns:
        A list of tuples: (fake_image_tensor, real_image_tensor, label_int)
    """
    base_dir = Path(base_dir)
    fake_dir = base_dir / "fake-images"
    real_dir = base_dir / "gt-images"
    label_dir = base_dir / "labels"

    # Find all .npy files in fake-images (assuming same names in other dirs)
    file_list = sorted(fake_dir.glob("*.npy"))
    samples = []

    for fpath in file_list:
        name = fpath.name

        # Load files
        fake_img = np.load(fake_dir / name)
        real_img = np.load(real_dir / name)
        label = np.load(label_dir / name).item()  # Load as int

        # Convert to tensors
        fake_tensor = torch.from_numpy(fake_img).float()
        real_tensor = torch.from_numpy(real_img).float()
        label_int = int(label)

        samples.append((fake_tensor, real_tensor, label_int))

    print(f"[✓] Loaded {len(samples)} samples from {base_dir.resolve()}")
    return samples



##################################################
# Get dataloader by group
##################################################
def get_dataloader_by_group(data_module, group: str):
    if group == "validation":
        return data_module.val_dataloader()
    elif group == "test":
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Unsupported group: {group}")
    




##################################################
# Run data collection and evaluation
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
    batch_size=32,
    max_samples=50000,
    num_steps=50,
    num_classes=1000,
    results_root="results",
    num_crops=4,
    crop_size=128,
    k=3,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)

    # Load model
    fm_module = TrainerModuleLatentFlow.load_from_checkpoint(checkpoint, map_location="cpu")
    fm_module.eval().to(device)

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
    dataloader = get_dataloader_by_group(data, group)

    # Setup results directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_results_dir / f"{model_name}_metrics.csv"

    # Loop over cfg scales
    with open(csv_path, "w") as csv_file:
        csv_file.write("cfg_scale,ccfg_scale,gFID,lFID,pFID,rFID,SSIM,LPIPS,PSNR,MSE,MAE\n")

        for cfg_scale, ccfg_scale in zip(cfg_scales, ccfg_scales):
            print(f"\n[INFO] CFG={cfg_scale}, CCFG={ccfg_scale}")

            # 1️ Collect and save samples
            sample_dir = base_results_dir / f"samples_cfg{cfg_scale}_ccfg{ccfg_scale}"
            collect_real_and_fake_features(
                fm_module=fm_module,
                data_loader=dataloader,
                output_root=sample_dir,
                source_timestep=source_timestep,
                max_samples=max_samples,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                num_steps=num_steps,
                num_classes=num_classes,
                device=device,
            )

            # 2️ Load saved samples as dataset
            sample_loader = load_saved_samples_as_dataset(sample_dir, batch_size=batch_size, shuffle=False)

            # 3️ Evaluate using trackers
            fid_tracker = FIDMetricsTracker(num_crops=num_crops, crop_size=crop_size, k=k, device=device)
            img_tracker = ImageMetricsTracker(device=device)

            for batch_fake, batch_real, _ in tqdm(sample_loader, desc=f"Evaluating CFG={cfg_scale}"):
                batch_fake = batch_fake.to(device)
                batch_real = batch_real.to(device)

                fid_tracker.update(batch_real, batch_fake)
                img_tracker.update(batch_real, batch_fake)

                # Memory cleanup
                del batch_fake, batch_real
                torch.cuda.empty_cache()
                gc.collect()


            # 4️ Aggregate and write metrics
            fid_metrics = fid_tracker.aggregate()
            img_metrics = img_tracker.aggregate()

            print(
                f"[INFO] gFID: {fid_metrics['gfid']:.4f} | "
                f"lFID: {fid_metrics['lfid'] if fid_metrics['lfid'] is not None else 'NA'} | "
                f"pFID: {fid_metrics['pFID']:.4f} | "
                f"rFID: {fid_metrics['rFID']:.4f} | "
                f"SSIM: {img_metrics['ssim']:.4f} | "
                f"LPIPS: {img_metrics['lpips']:.4f} | "
                f"PSNR: {img_metrics['psnr']:.2f} | "
                f"MSE: {img_metrics['mse']:.6f} | "
                f"MAE: {img_metrics['mae']:.6f}"
            )

            csv_file.write(
                f"{cfg_scale},{ccfg_scale},"
                f"{fid_metrics['gfid']:.6f},"
                f"{fid_metrics['lfid'] if fid_metrics['lfid'] is not None else 'NA'},"
                f"{fid_metrics['pFID']:.6f},"
                f"{fid_metrics['rFID']:.6f},"
                f"{img_metrics['ssim']:.6f},"
                f"{img_metrics['lpips']:.6f},"
                f"{img_metrics['psnr']:.6f},"
                f"{img_metrics['mse']:.6f},"
                f"{img_metrics['mae']:.6f}\n"
            )
            csv_file.flush()

            # Clean up
            del fid_tracker, img_tracker, sample_loader
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nDone! Metrics saved to: {csv_path}")




##################################################
# Example usage
##################################################

if __name__ == "__main__":

    #####################################
    # Evaluation Setup
    #####################################
    source_timestep     = 0.50
    target_timestep     = 1.00
    beta                = 0.1
    dataset_name        = 'imagenet256-dataset-T000006'
    test_dataset_name   = 'imagenet256-testset-T151412'
    group               = "validation"  # or "test"
    baseline            = (source_timestep == 0.50 and target_timestep == 0.50)


    num_crops           = 2
    crop_size           = 64
    k                   = 3

    #####################################
    # Model Paths for SiT-XL-2
    #####################################
    # Example: pick your checkpoint depending on your experiment
    # beta: 0.1
    DiTSXL_Beta05x05x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt'   ### (Baseline)
    DiTSXL_Beta00x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA02x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_2.0b/BetaVAE-B-2/V0/2025-07-10/30912/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA05x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt'  
    
    # beta: 1.0
    DITSXL_Beta05x05x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt'    
    DITSXL_Beta02x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt'   
    DITSXL_Beta05x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt'          

    # beta: 5.0
    DiTSXL_Beta05x05x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt'
    DiTSXL_Beta02x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt'    
    DiTSXL_Beta05x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt'      
    
    
    # Pick your experiment
    checkpoint = DITSXL_BETA05x10x_01b 

    #####################################
    # Dataset & Evaluation Parameters
    #####################################
    validation_data_path = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5'
    test_data_path       = './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'
    data_path            = validation_data_path if group == "validation" else test_data_path

    project_name         = "CFM_Quantitative_Eval_Baseline" if baseline else "CFM_Quantitative_Eval"
    model_name           = f"Beta-VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"

    #####################################
    # Device + Seed Setup
    #####################################
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #####################################
    # Evaluation Parameters
    #####################################
    max_samples = 30000     # Maximum number of samples to collect (50k is standard practice for FID)
    batch_size  = 24         # Batch size for evaluation (smaller, e.g., 16 or 32)
    num_steps   = 50        # Fixed number of reverse-time steps for generation
    num_classes = 1000
    cfg_scales  = [1.0, 3.0] # 5.0, 7.0, 9.0
    ccfg_scales = [1.0, 1.0] # 1.0, 1.0, 1.0

    #####################################
    # Run unified collection and evaluation
    #####################################
    run_data_collection_and_evaluation(
        checkpoint=checkpoint,
        data_path=data_path,
        project_name=project_name,
        model_name=model_name,
        group=group,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        cfg_scales=cfg_scales,
        ccfg_scales=ccfg_scales,
        batch_size=batch_size,
        max_samples=max_samples,
        num_steps=num_steps,
        num_classes=num_classes,
        results_root="results",
        num_crops=num_crops,
        crop_size=crop_size,
        k=k,
    )
    
    

# CUDA_VISIBLE_DEVICES=2 python ...
