
import logging
from typing import Dict, Union
import numpy as np
import torch


# -----------------------------------------------------------------
# Reference:
# https://github.com/yukimasano/linear-probes/blob/master/files.py
# -----------------------------------------------------------------

class MovingAverage():
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# -----------------------------------------------------------------
# Reference: Reconstruction loss
# -----------------------------------------------------------------

def normalize_images(images: Union[torch.Tensor, np.ndarray], min_val=None, max_val=None) -> torch.Tensor:
    """Normalize images to the range [0, 1]."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    images = images.float()
    eps = 1e-10 # Avoid zero division

    if min_val is None or max_val is None:
        dims = tuple(range(1, images.ndim))
        min_val = images.amin(dims, keepdim=True) if min_val is None else min_val
        max_val = images.amax(dims, keepdim=True) if max_val is None else max_val
    
    # Normalize and clip values
    denominator = max_val - min_val + eps
    images = torch.clamp((images - min_val) / denominator, 0.0, 1.0)
    
    return images


def reconstruction_error(real_ims: Union[torch.Tensor, np.ndarray], fake_ims: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
   """Compute reconstruction error metrics. Expects images in the range [0, 1]."""
   # Convert inputs to torch tensors on CPU
   real_ims = real_ims.detach().cpu() if isinstance(real_ims, torch.Tensor) else torch.from_numpy(real_ims)
   fake_ims = fake_ims.detach().cpu() if isinstance(fake_ims, torch.Tensor) else torch.from_numpy(fake_ims)

   if real_ims.shape != fake_ims.shape:
       raise ValueError(f"Shape mismatch: real {real_ims.shape} vs fake {fake_ims.shape}")
   
   # Normalize images to [0,1] range
   real_ims = normalize_images(real_ims)
   fake_ims = normalize_images(fake_ims)

   # Compute MSE (keeping batch dimension)
   mse = torch.mean((real_ims - fake_ims) ** 2, dim=tuple(range(1, real_ims.ndim)))
   
   # Compute PSNR (avoiding log of zero)
   psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

   # Return averaged metrics
   return {
       'mse': mse.mean().item(),
       'psnr': psnr.mean().item()
   }

# -----------------------------------------------------------------
# Reference: Model Training
# -----------------------------------------------------------------

"""EMA (Exponential Moving Average) Metric Tracker"""
class EMATracker:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, value):
        # Ensure value is a float
        if isinstance(value, (list, tuple)):
            value = value[0] 
        value = float(value)
        
        if self.value is None:
            self.value = value
            return value
        
        self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value