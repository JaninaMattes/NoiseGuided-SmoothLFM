import torch
import torch.nn as nn


""" Metrics for evaluating image singal quality. """

class PSNR(nn.Module):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: (n, c, h, w) - input image 1
        img2: (n, c, h, w) - input image 2
        v_max: maximum pixel value in the image (usually 255 for 8-bit images)
        eps: small value added for numerical stability
    """
    def __init__(self, v_max=255., eps=1e-10):
        super().__init__()
        self.v_max = v_max
        self.eps = eps

    def forward(self, img1, img2):
        # Calculate MSE + PSNR
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(self.v_max / torch.sqrt(mse + self.eps))

        return psnr.mean()


class MSE(nn.Module):
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        img1: (n, c, h, w) - input image 1
        img2: (n, c, h, w) - input image 2
    """
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2):
        # Calculate MSE
        return torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])