# Code adapted from:
# - https://github.com/CompVis/fm-boosting/blob/main/fmboost/metrics.py
import torch
import torch.nn as nn

import torchvision.transforms.functional as F
from torchvision import transforms

# helper 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from ldm.helpers import un_normalize_ims
from ldm.helpers import denorm_tensor, un_normalize_ims


def un_normalize_ims(ims):
    """ Convert from [-1, 1] or [0, 1] to [0, 255] """
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
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnrs = 20 * torch.log10(1 / torch.sqrt(mse + 1e-8))
    return psnrs.mean()


""" Image metrics tracker """


class ImageMetricsTracker(nn.Module):
    def __init__(self, num_crops: int = 4, crop_size: int = 128):
        super().__init__()
        self.ssim = SSIM(data_range=1.0)    # SSIM - requires [0, 1] range
        self.psnr = PSNR(data_range=1.0)    # PSNR - requires [0, 1] range
        self.mse = nn.MSELoss()                            
        self.lpips = LPIPS(net_type='alex') # expects pixel values in [-1, 1]   

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )

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

                    croped_real.append(F.crop(img_real, *anchor))
                    croped_fake.append(F.crop(img_fake, *anchor))
            
            real_ims = torch.stack(croped_real)
            fake_ims = torch.stack(croped_fake)
            
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        # Normalize: [0, 255] -> [0, 1]   
        pred = denorm_tensor(pred, min=0, max=1, keep_channels=3).float()                                       
        target = denorm_tensor(target, min=0, max=1, keep_channels=3).float()
        
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
        fid = self.fid.compute()
        ssim = torch.stack(self.ssims).mean()
        psnr = torch.stack(self.psnrs).mean()
        mse = torch.stack(self.mses).mean()
        mae = torch.stack(self.maes).mean()
        lpips = torch.stack(self.lpips_scores).mean()
        out = dict(fid=fid, ssim=ssim, psnr=psnr, mse=mse, mae=mae, lpips=lpips)
        return out