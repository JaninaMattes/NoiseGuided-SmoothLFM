# Code adapted from:
# - https://github.com/CompVis/fm-boosting/blob/main/fmboost/metrics.py
import cv2
import numpy as np

from einops import rearrange
import torch
import torch.nn as nn

import torchvision.transforms.functional as F
from torchvision import transforms

# Measzres
import pywt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA

# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional.pairwise import pairwise_cosine_similarity as PCS

from ldm.helpers import denorm_tensor, un_normalize_ims


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims

def normalize_ims(ims):
    """ Convert from [0, 255] to [0, 1] """
    ims = ims.float() / 255.
    return ims

def normalize_ims_one_minus(ims):
    """ Convert from [0, 255] to [-1, 1] """
    ims = ims.float() / 127.5 - 1
    return ims

def calculate_PSNR(img1, img2):
    """ Calculate PSNR between two imgs in [0, 1] range """
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnrs = 20 * torch.log10(1 / torch.sqrt(mse + 1e-8))
    return psnrs.mean()

def calculate_SNR(signal, noisy_signal):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB between target and prediction.
    """
    signal = torch.clamp(signal, 0, 1)
    noisy_signal = torch.clamp(noisy_signal, 0, 1)

    noise = noisy_signal - signal
    signal_power = torch.mean(signal ** 2, dim=[1, 2, 3])
    noise_power = torch.mean(noise ** 2, dim=[1, 2, 3])
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.mean()

def calculate_CosineSim(img1, img2):
    """
    Calculate average cosine similarity between two batches of images.
    """
    img1 = img1.float()
    img2 = img2.float()
    B = img1.shape[0]
    cosine_sim = PCS(img1.reshape(B, -1), img2.reshape(B, -1))
    return cosine_sim.mean()

    
def analyze_SVD(img):
    """
    Returns:
        sv_list: list of singular values per channel
        sv_energy: list of normalized energy for each channel
    """
    C = img.shape[0]
    sv_list = []
    sv_energy = []

    for c in range(C):
        mat = img[c]  # Shape: (H, W)
        U, S, Vh = torch.linalg.svd(mat)
        sv_list.append(S.cpu())

        # Normalize to total energy
        energy = (S**2).sum()
        energy_ratio = (S**2 / energy).cpu()
        sv_energy.append(energy_ratio)

    return sv_list, sv_energy

def svd_reconstruction_rmse(mat, k):
    U, S, Vh = torch.linalg.svd(mat)
    S_truncated = torch.zeros_like(S)
    S_truncated[:k] = S[:k]
    recon = (U @ torch.diag(S_truncated) @ Vh)
    rmse = torch.sqrt(((mat - recon) ** 2).mean())
    return rmse

def compute_noise_metric(img, top_k=10):
    rmse_list = []
    for c in range(img.shape[0]):
        rmse = svd_reconstruction_rmse(img[c], k=top_k)
        rmse_list.append(rmse.item())
    return sum(rmse_list) / len(rmse_list)


def svd_energy(img, k=None):
    """ Measures information content of the img using SVD """
    img = rearrange(img, 'b c h w -> (b c) h w')  # (B*C, H, W)
    energies = [
        (s[:k] ** 2).sum() / (s ** 2).sum() if k is not None else 1.0
        for s in [torch.linalg.svd(mat, full_matrices=False).S for mat in img]
    ]
    return torch.tensor(energies, device=img.device).mean()


def calculate_entropy(img, bins=256):
    """
    Calculate entropy of the img.
    """
    img = img.float().view(-1)  # flatten the img
    hist = torch.histc(img, bins=bins, min=0.0, max=1.0)
    hist = hist / (hist.sum() + 1e-8)
    entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
    return entropy.item()



# https://dsp.stackexchange.com/questions/102/how-do-you-measure-detail-of-a-signal
# Detect edges in the img using Laplacian filter

def edge_sharpness(img):
    """
    Calculate the sharpness of an img using Laplacian variance.
    """
    gray = img.mean(dim=0).cpu().numpy()
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def calculate_kurtosis(img):
    """ Large kurtosis indicates a high peak and heavy tails """
    img = img.cpu().numpy()
    return kurtosis(img.flatten())

def wavelet_kurtosis_map(img, wavelet='db1'):
    coeffs2 = pywt.dwt2(img.cpu().numpy(), wavelet)
    cA, (cH, cV, cD) = coeffs2
    return {
        'cH_kurt': kurtosis(cH.flatten()),
        'cV_kurt': kurtosis(cV.flatten()),
        'cD_kurt': kurtosis(cD.flatten()),
    }

def pca_signal_strength(tensor, n_components=10):
    """
    tensor: torch.Tensor of shape (C, H, W)
    returns: signal_strength, effective_rank, condition_number
    """
    C, H, W = tensor.shape
    flat = tensor.view(C, -1).cpu().numpy().T  # Shape: (H*W, C)

    pca = PCA()
    pca.fit(flat)

    evr = pca.explained_variance_ratio_
    signal_strength = evr[:n_components].sum()

    # Effective rank (entropy of explained variance)
    entropy = -np.sum(evr * np.log(evr + 1e-12))
    effective_rank = np.exp(entropy)

    # Condition number as ratio of singular values
    cond_number = pca.singular_values_[0] / (pca.singular_values_[-1] + 1e-8)

    return {
        "signal_strength": signal_strength,
        "effective_rank": effective_rank,
        "condition_number": cond_number
    }




""" Latent Sample Noise Metrics Tracker """

class NoiseMetricsTracker(nn.Module):
    def __init__(self, num_crops: int = 4, crop_size: int = 512):
        super().__init__()
        self.ssim = SSIM(data_range=1.0)    # SSIM - requires [0, 1] range
        self.psnr = PSNR(data_range=1.0)    # PSNR - requires [0, 1] range
        self.mse = nn.MSELoss()                                               

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )

        # whether use the fid on crops during training
        self.patch_fid = num_crops > 0
        if self.patch_fid:
            print("[imgMetricTracker] Evaluating using patch-wise FID")
        self.num_crops = num_crops
        self.crop_size = crop_size

        # initialize
        self.reset()
        

    def __call__(self, img_target, img_pred, noise_target=None, noise_pred=None):
        """ 
        Assumes target and pred in discretized range [0, 255]. 
        If in [0, 1] range, it will be converted to [0, 255].
        """
        # Data range [-1, 1] -> [0, 255] 
        real_ims = un_normalize_ims(img_target) if img_target.max() <= 1 else img_target
        fake_ims = un_normalize_ims(img_pred) if img_pred.max() <= 1 else img_pred

        ##################
        #   Pixel Space  #
        ##################
        
        # FID  
        if self.patch_fid:
            croped_real = []
            croped_fake = []
            anchors = []
            for i in range(real_ims.shape[0] * self.num_crops):
                anchors.append(transforms.RandomCrop.get_params(
                    real_ims[0], output_size=(self.crop_size, self.crop_size)))

            for idx, (img_real, img_fake) in enumerate(zip(real_ims, fake_ims)):
                for i in range(self.num_crops):
                    anchor = anchors[idx * self.num_crops + i]
                    croped_real.append(F.crop(img_real, *anchor))
                    croped_fake.append(F.crop(img_fake, *anchor))

            real_ims = torch.stack(croped_real)
            fake_ims = torch.stack(croped_fake)

        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        ##################
        #   Latent Space #
        ##################
        if noise_target is not None and noise_pred is not None:
            target = denorm_tensor(noise_target, min=0, max=1, keep_channels=3).float()  # 3 channels
            pred = denorm_tensor(noise_pred, min=0, max=1, keep_channels=3).float()      # 3 channels

            # Compute metrics over the whole image (3-channels)
            self.cosine_sims.append(calculate_CosineSim(pred, target))# Cosine Similarity
            self.ssims.append(self.ssim(pred, target))                # SSIM
            self.psnrs.append(self.psnr(pred, target))                # PSNR
            self.snrs.append(calculate_SNR(target, pred))             # SNR
            self.mses.append(torch.mean((pred - target) ** 2, dim=[1, 2, 3]))  # MSE


    def reset(self):
        # In pixel space
        self.fid.reset()
        # In latent space
        self.cosine_sims = []
        self.ssims = []
        self.psnrs = []
        self.snrs = []
        self.mses = []


    def aggregate(self):
        """ Compute per batch metrics """
        # FID for image base feature
        fid = self.fid.compute()
        # Noise metrics
        cosine_sim = torch.stack(self.cosine_sims).mean()
        ssim = torch.stack(self.ssims).mean()
        psnr = torch.stack(self.psnrs).mean()
        snr = torch.stack(self.snrs).mean()
        mse = torch.stack(self.mses).mean()
        
        # Output 
        out = dict(
            fid=fid, ssim=ssim, psnr=psnr, snr=snr, mse=mse, cosine_sim=cosine_sim
        )
        return out