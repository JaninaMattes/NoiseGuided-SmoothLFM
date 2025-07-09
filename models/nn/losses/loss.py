# Adapted code from: 
# - https://github.com/lucidrains/rectified-flow-pytorch/blob/main/rectified_flow_pytorch/rectified_flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models import VGG16_Weights


""" Utility functions. """


def exists(val):
    return val is not None


def default(val, d):
    return val if d is None else d



""" Loss functions. """



# -------------------------
class MSE(nn.Module):
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        img1: (n, c, h, w) - input image 1
        img2: (n, c, h, w) - input image 2
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Calculate MSE
        return F.mse_loss(pred, target)
    
    
# -------------------------
class LPIPSLoss(nn.Module):
    def __init__(
        self,
        vgg: nn.Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss


class PseudoHuberLoss(nn.Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss
    

class PseudoHuberLossWithLPIPS(nn.Module):
    def __init__(self, data_dim: int = 3, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, pred_data, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-1))
        return time_weighted_loss.mean()


# -------------------------
class SpectralLoss(nn.Module):
    """Compute sum of the squared frequency amplitudes.
        idea: https://bartwronski.com/2021/07/06/comparing-images-in-frequency-domain-spectral-loss-does-it-make-sense/
    """
    def __init__(self, normalize=True, eps=1e-10):
        self.normalize = normalize
        self.eps = eps                                          # Small value to avoid division by zero
    
    def forward(self, img1, img2):
        fft_1 = torch.fft.fft2(img1, dim=(-2, -1))              # Apply FFT on height and width dimensions
        fft_2 = torch.fft.fft2(img2, dim=(-2, -1))
        
        amplitude_1 = torch.abs(fft_1)
        amplitude_2 = torch.abs(fft_2)
        
        if self.normalize:
            # Normalize by maximum amplitude per channel
            amplitude_1 = amplitude_1 / (amplitude_1.amax(dim=(-2, -1), keepdim=True) + self.eps)
            amplitude_2 = amplitude_2 / (amplitude_2.amax(dim=(-2, -1), keepdim=True) + self.eps)
            
        return F.mse_loss(amplitude_1, amplitude_2)



# -------------------------
class MMDLoss(nn.Module):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
        
        taken from: https://github.com/yiftachbeer/mmd_loss_pytorch/
                    https://arxiv.org/abs/1502.02761
        
    """
    def __init__(self, kernel="multiscale", device="cuda"):
        self.kernel = kernel
        self.device = device
        
    def forward(self, x, y):
        # Flatten the spatial and channel dimensions
        x_flat = x.reshape(x.shape[0], -1)  # (B, C*H*W)
        y_flat = y.reshape(y.shape[0], -1)  # (B, C*H*W)
        
        # Compute kernel matrices
        xx = torch.mm(x_flat, x_flat.t())  # (B, B)
        yy = torch.mm(y_flat, y_flat.t())  # (B, B)
        zz = torch.mm(x_flat, y_flat.t())  # (B, B)
        
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        
        XX = torch.zeros(xx.shape).to(self.device)
        YY = torch.zeros(xx.shape).to(self.device)
        XY = torch.zeros(xx.shape).to(self.device)
        
        if self.kernel == "multiscale":
            # Multiple bandwidth parameters for better coverage
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        elif self.kernel == "rbf":
            # Scale bandwidth with data dimensionality
            data_dim = x_flat.shape[1]  # C*H*W
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/(a * data_dim))
                YY += torch.exp(-0.5*dyy/(a * data_dim))
                XY += torch.exp(-0.5*dxy/(a * data_dim))
        
        return torch.mean(XX + YY - 2. * XY)
    


if __name__ == "__main__":
    
    # Test Spectral Loss
    img1 = torch.randn(16, 4, 32, 32)
    img2 = torch.randn(16, 4, 32, 32)
    loss = SpectralLoss(normalize=True)
    spec_loss = loss(img1, img2)
    print(spec_loss.item())