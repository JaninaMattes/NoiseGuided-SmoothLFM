import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft



class FrequencyUnit(nn.Module):
    def __init__(self, low_limit=0.1, high_limit=0.2):
        """
        Splits an image into low-frequency and high-frequency components for multi-channel input.
        inspired by: https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
        
        Pytorch doc: https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/
        """
        super().__init__()
        self.low_limit = low_limit
        self.high_limit = high_limit

    
    def lowpass_filter(self, input):
        """
        Applies a low-pass filter independently for each channel.
        """
        B, C, H, W = input.shape
        low_filtered = []
        for c in range(C):
            channel = input[:, c, :, :]  # Extract channel (B, H, W)
            pass1 = torch.abs(fft.rfftfreq(W, d=1.0)) < self.low_limit
            pass2 = torch.abs(fft.fftfreq(H, d=1.0)) < self.low_limit
            kernel = torch.outer(pass2, pass1).to(input.device)
            fft_channel = fft.rfft2(channel)
            filtered = fft.irfft2(fft_channel * kernel, s=(H, W))
            low_filtered.append(filtered.unsqueeze(1))  # Add channel dimension back
        return torch.cat(low_filtered, dim=1)  # Combine filtered channels

    
    
    def highpass_filter(self, input):
        """
        Applies a high-pass filter, handling both patch tokens and regular feature maps.
        
        Args:
            input: Either patch tokens (B, num_patches, embed_dim) or feature maps (B, C, H, W)
        """
        if len(input.shape) == 3:                                       # Patch tokens (B, num_patches, embed_dim)
            B, num_patches, embed_dim = input.shape
            
            H = W = int(np.sqrt(num_patches))
            if H * W != num_patches:
                raise ValueError(f"Number of patches {num_patches} must be a perfect square")
                
            # Reshape to (B, embed_dim, H, W) for spatial filtering
            x = input.transpose(1, 2).reshape(B, embed_dim, H, W)
            
            high_filtered = []
            for c in range(embed_dim):
                channel = x[:, c, :, :]  # Extract channel (B, H, W)
                pass1 = torch.abs(fft.rfftfreq(W, d=1.0)) >= self.high_limit
                pass2 = torch.abs(fft.fftfreq(H, d=1.0)) >= self.high_limit
                kernel = torch.outer(pass2, pass1).to(input.device)
                fft_channel = fft.rfft2(channel)
                filtered = fft.irfft2(fft_channel * kernel, s=(H, W))
                high_filtered.append(filtered.unsqueeze(1))
            
            # Reconstruct the tensor
            filtered = torch.cat(high_filtered, dim=1)                      # (B, embed_dim, H, W)
            return filtered.reshape(B, embed_dim, -1).transpose(1, 2)       # (B, num_patches, embed_dim)
            
        else:                                                               # Regular feature maps (B, C, H, W)
            B, C, H, W = input.shape
            high_filtered = []
            for c in range(C):
                channel = input[:, c, :, :]                                 # Extract channel (B, H, W)
                pass1 = torch.abs(fft.rfftfreq(W, d=1.0)) >= self.high_limit
                pass2 = torch.abs(fft.fftfreq(H, d=1.0)) >= self.high_limit
                kernel = torch.outer(pass2, pass1).to(input.device)
                fft_channel = fft.rfft2(channel)
                filtered = fft.irfft2(fft_channel * kernel, s=(H, W))
                high_filtered.append(filtered.unsqueeze(1))
            return torch.cat(high_filtered, dim=1)


    def forward(self, x):
        """
        Splits the input tensor into low-frequency and high-frequency components for all channels.
        """
        low_freq = self.lowpass_filter(x)
        high_freq = self.highpass_filter(x)
        return low_freq, high_freq
