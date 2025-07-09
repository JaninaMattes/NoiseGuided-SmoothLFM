# Code adpated from:
# - https://github.com/forever208/DCTdiff/blob/DCTdiff/datasets.py#L392
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import img
import os
import glob
import einops
import torchvision.transforms.functional as F
import cv2


import torch
import torch.nn as nn

from ldm.models.dct_condition.dct.torch_dct import dct_2d, idct_2d

""" Prepare dataset """

def center_crop_arr(pil_img, img_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_img.size) >= 2 * img_size:
        pil_img = pil_img.resize(
            tuple(x // 2 for x in pil_img.size), resample=img.BOX
        )

    scale = img_size / min(*pil_img.size)
    pil_img = pil_img.resize(
        tuple(round(x * scale) for x in pil_img.size), resample=img.BICUBIC
    )

    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - img_size) // 2
    crop_x = (arr.shape[1] - img_size) // 2
    return arr[crop_y : crop_y + img_size, crop_x : crop_x + img_size]


def random_crop_arr(pil_img, img_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(img_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(img_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_img.size) >= 2 * smaller_dim_size:
        pil_img = pil_img.resize(
            tuple(x // 2 for x in pil_img.size), resample=img.BOX
        )

    scale = smaller_dim_size / min(*pil_img.size)
    pil_img = pil_img.resize(
        tuple(round(x * scale) for x in pil_img.size), resample=img.BICUBIC
    )

    arr = np.array(pil_img)
    crop_y = random.randrange(arr.shape[0] - img_size + 1)
    crop_x = random.randrange(arr.shape[1] - img_size + 1)
    return arr[crop_y: crop_y + img_size, crop_x: crop_x + img_size]


""" DCT functions """

def split_into_blocks(img, block_sz):
    blocks = []
    for i in range(0, img.shape[0], block_sz):
        for j in range(0, img.shape[1], block_sz):
            blocks.append(img[i:i + block_sz, j:j + block_sz])  # first row, then column
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    img = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            img[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return img

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) - 128  # Shift to center around 0
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block + 128  # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)


""" DCT to Image """


class DCTEmbedder:
    def __init__(self, norm=None, remove_low_freq_ratio=0.5, height=32, width=32, block_size=8):
        super(DCTEmbedder, self).__init__()
        self.norm = norm
        self.remove_low_freq_ratio = remove_low_freq_ratio
        self.block_size = block_size
        self.height = height
        self.width = width

    def encode(self, x):
        """
        Perform the DCT transform, remove lower frequencies, and then apply the IDCT.
        Args:
            x (torch.Tensor): Latent tensor of shape (B, 4, 32, 32)
        Returns:
            torch.Tensor: Reconstructed tensor after removing lower frequencies
        """
        
        # Step 1: Split into blocks
        x_blocks = self.split_into_blocks(x, self.block_size)           # (B, 4, num_blocks, 8, 8)
        B_blocks = x_blocks.view(-1, self.block_size, self.block_size)  # (B * num_blocks*4, 8, 8)

        # Step 2: Apply DCT on each block
        B_dct_blocks = self.dct_2d(B_blocks)                              # (B * num_blocks*4, 8, 8)
        
        # Step 3: Sort frequencies by magnitude 
        # and remove low frequencies
        dct_tokens = self.remove_low_frequencies(B_dct_blocks)         # (B * num_blocks*4, 8, 8)
        return dct_tokens
    
    
    def decode(self, tokens):
        """
        Reconstruct the image from the DCT coefficients.
        Args:
            tokens (torch.Tensor): DCT coefficients
        Returns:
            torch.Tensor: Reconstructed image
        """
        
        # Step 1: Apply IDCT on each block
        x_blocks = self.idct_2d(tokens)                                   # (B * num_blocks*4, 8, 8)
        
        # Step 2: Combine blocks
        x_recon = self.combine_blocks(x_blocks, self.block_size, self.height, self.width)
        return x_recon
    

    def dct_2d(self, x):
        """
        2D Discrete Cosine Transform (DCT-II).
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: DCT of input
        """
        X1 = self.dct(x)
        X2 = self.dct(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)

    def idct_2d(self, X):
        """
        2D Inverse Discrete Cosine Transform (IDCT-III).
        Args:
            X (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: IDCT of input
        """
        x1 = self.idct(X)
        x2 = self.idct(x1.transpose(-1, -2))
        return x2.transpose(-1, -2)

    def dct(self, x):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT).
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: DCT-II of input
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = self.dct_fft_impl(v)

        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        if self.norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        return 2 * V.view(*x_shape)

    def idct(self, X):
        """
        Inverse Discrete Cosine Transform, Type III (a.k.a. the IDCT).
        Args:
            X (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: IDCT-II of input
        """
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if self.norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = self.idct_irfft_impl(V)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)


    def dct_fft_impl(self, v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(self, V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


    def remove_low_frequencies(self, x_dct):
        """
        Removes the lower frequencies by zeroing out the smallest magnitudes.
        Args:
            x_dct (torch.Tensor): DCT coefficients
        Returns:
            torch.Tensor: DCT coefficients with low frequencies removed
        """
        # Compute the magnitude
        mag = torch.abs(x_dct)
        
        # Flatten the magnitudes and sort them
        mag_flat = mag.view(mag.size(0), -1)
        threshold_idx = int(mag_flat.size(1) * self.remove_low_freq_ratio)
        
        # Keep the threshold value
        threshold = torch.topk(mag_flat, threshold_idx, dim=-1, largest=False).values[:, -1].unsqueeze(-1).unsqueeze(-1)
        
        # Mask to remove zero out the lower frequencies
        mask = mag >= threshold
        x_dct = x_dct * mask.float()

        return x_dct

    def split_into_blocks(self, x, block_sz):
        B, C, H, W = x.shape
        x = x.unfold(2, block_sz, block_sz).unfold(3, block_sz, block_sz)  # (B, C, H//bs, W//bs, bs, bs)
        x = x.contiguous().view(B, C, -1, block_sz, block_sz)  # (B, C, num_blocks, bs, bs)
        return x

    def combine_blocks(self, blocks, block_sz, H, W):
        B, C, num_blocks, bs, _ = blocks.shape
        blocks = blocks.view(B, C, H // bs, W // bs, bs, bs)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        blocks = blocks.view(B, C, H, W)
        return blocks




class DCTPatchEmbedder(nn.Module):
    """ DCT Embedder for image patches.
        projects the patch_size * patch_size * in_channels blocks 
        into frequency tokens (i.e. N coefficients per batch).
    """
    
    def __init__(self, patch_size=4, in_chans=4, embed_dim=768, dct_norm='ortho', high_freqs=None, low2high_order=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dct_norm = dct_norm

        self.block_dim = patch_size * patch_size * in_chans
        self.high_freqs = high_freqs                # number of high-frequency coeffs to keep
        self.low2high_order = low2high_order        # custom zig-zag or spectral ordering

        if high_freqs is not None:
            proj_in_dim = high_freqs * in_chans
        else:
            proj_in_dim = self.block_dim
        self.proj = nn.Linear(proj_in_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected {self.in_chans} channels but got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        # Step 1: Split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                    .unfold(3, self.patch_size, self.patch_size)                        # (B, C, H//p, W//p, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)                                     # (B, H//p, W//p, C, p, p)
        patches = patches.reshape(B, -1, C, self.patch_size, self.patch_size)           # (B, N, C, p, p)

        # Step 2: Batch DCT on all channels
        patches = patches.reshape(-1, self.patch_size, self.patch_size)                 # (B*N*C, p, p)
        dct_patches = dct_2d(patches, norm=self.dct_norm)                               # (B*N*C, p, p)
        dct_patches = dct_patches.reshape(B, -1, C, self.patch_size * self.patch_size)  # (B, N, C, p*p)

        # Step 3: Óptional reorder
        if self.low2high_order is not None:
            dct_patches = dct_patches[:, :, :, self.low2high_order]                     # (B, N, C, p*p)

        # Step 4: Slice high-freqs if applicable
        if self.high_freqs is not None:
            dct_patches = dct_patches[:, :, :, -self.high_freqs:]                       # (B, N, C, high_freqs)

        # Step 5: Flatten across channels
        dct_patches = dct_patches.flatten(2, 3)                                         # (B, N, C * high_freqs)

        # Step 6: Linear projection
        tokens = self.proj(dct_patches)                                                 # (B, N, embed_dim)
        return tokens




class IDCTPatchDecoder(nn.Module):
    """Inverse of DCTPatchEmbed - reconstruct spatial patches from frequency domain."""
    def __init__(self, patch_size=4, out_channels=4, embed_dim=768, dct_norm='ortho', high_freqs=None, low2high_order=None):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.dct_norm = dct_norm
        self.high_freqs = high_freqs
        self.low2high_order = low2high_order

        # Inverse linear layer (decoder)
        in_dim = out_channels * (high_freqs if high_freqs is not None else patch_size * patch_size)
        self.proj = nn.Linear(embed_dim, in_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, embed_dim) - from FinalLayer or decoder output.
        Returns:
            (B, N, out_channels, patch_size, patch_size) - spatial patches.
        """
        B, N, _ = x.shape
        p = self.patch_size
        C = self.out_channels

        # Step 1: Inverse projection
        x = self.proj(x)  # (B, N, C * high_freqs)

        # Step 2: Unflatten channel-frequency
        if self.high_freqs is not None:
            x = x.view(B, N, C, self.high_freqs)  # (B, N, C, high_freqs)
        else:
            x = x.view(B, N, C, p * p)            # (B, N, C, p*p)

        # Step 3: Inverse reorder if applicable
        if self.low2high_order is not None:
            inv_order = torch.argsort(torch.tensor(self.low2high_order, device=x.device))
            full_spectrum = torch.zeros(B, N, C, p*p, device=x.device, dtype=x.dtype)
            full_spectrum[:, :, :, inv_order[-self.high_freqs:]] = x  
            x = full_spectrum  # (B, N, C, p*p)
        elif self.high_freqs is not None:
            pad_size = p * p - self.high_freqs
            x = F.pad(x, (pad_size, 0))                 # pad before high-freqs

        # Step 4: Unflatten (B, N, C, p, p)
        x = x.view(B, N, C, p, p)

        # Step 5: Apply iDCT2D patch-wise
        x = x.view(-1, p, p)                          # (B*N*C, p, p)
        x = idct_2d(x, norm=self.dct_norm)            # (B*N*C, p, p)
        x = x.view(B, N, C, p, p)                     # (B, N, C, p, p)

        return x



    
class DCT_4YCbCr:
    def __init__(self, root_dir, img_sz=64, tokens=0, low_freqs=0, block_sz=8, low2high_order=None, reverse_order=None,
                 Y_bound=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.img_paths.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

        # parameters of DCT design
        self.Y_bound = np.array(Y_bound)
        print(f"using Y_bound {self.Y_bound} for training")
        self.tokens = tokens
        self.low_freqs = low_freqs
        self.block_sz = block_sz

        Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
        self.Y_blocks_per_row = int(img_sz / block_sz)
        self.index = []  # index of Y if merging 2*2 Y-block area
        for row in range(0, Y, int(2 * self.Y_blocks_per_row)):  # 0, 32, 64...
            for col in range(0, self.Y_blocks_per_row, 2):  # 0, 2, 4...
                self.index.append(row + col)
        assert len(self.index) == int(Y / 4)

        self.low2high_order = low2high_order
        self.reverse_order = reverse_order

        # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
        self.cb_index = [i for i in range(4, tokens, 6)]
        self.cr_index = [i for i in range(5, tokens, 6)]
        self.y_index = [i for i in range(0, tokens) if i not in self.cb_index and i not in self.cr_index]
        assert len(self.y_index) + len(self.cb_index) + len(self.cr_index) == tokens

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = img.open(img_path).convert('RGB')
        # img.save('original_img.jpg')
        img = transforms.RandomHorizontalFlip()(img)  # do data augmentation by PIL
        img = np.array(img)

        # Step 1: Convert RGB to YCbCr
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img_y = 0.299 * R + 0.587 * G + 0.114 * B
        img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)
        cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 2: Split the Y, Cb, and Cr components into 4x4 blocks
        y_blocks = split_into_blocks(img_y, self.block_sz)  # Y component, (64, 64) --> (256, 4, 4)
        cb_blocks = split_into_blocks(cb_downsampled, self.block_sz)  # Cb component, (32, 32) --> (64, 4, 4)
        cr_blocks = split_into_blocks(cr_downsampled, self.block_sz)  # Cr component, (32, 32) --> (64, 4, 4)

        # Step 3: Apply DCT on each block
        dct_y_blocks = dct_transform(y_blocks)  # (256, 4, 4)
        dct_cb_blocks = dct_transform(cb_blocks)  # (64, 4, 4)
        dct_cr_blocks = dct_transform(cr_blocks)  # (64, 4, 4)

        # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks pixel region)
        DCT_blocks = []
        for i in range(dct_cr_blocks.shape[0]):
            DCT_blocks.append([
                dct_y_blocks[self.index[i]],  # Y
                dct_y_blocks[self.index[i] + 1],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row],  # Y
                dct_y_blocks[self.index[i] + self.Y_blocks_per_row + 1],  # Y
                dct_cb_blocks[i],  # Cb
                dct_cr_blocks[i],  # Cr
            ])
        DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, self.block_sz*self.block_sz)  # (64, 6, 4, 4) --> (64, 6, 16)

        # Step 5: scale into [-1, 1]
        assert DCT_blocks.shape == (self.tokens, 6, self.block_sz*self.block_sz)
        DCT_blocks[:, :4 :] = (DCT_blocks[:, :4 :]) / self.Y_bound
        DCT_blocks[:, 4, :] = (DCT_blocks[:, 4, :]) / self.Y_bound
        DCT_blocks[:, 5, :] = (DCT_blocks[:, 5, :]) / self.Y_bound

        # Step 6: reorder coe from low to high freq, then mask out high-freq signals
        DCT_blocks = DCT_blocks[:, :, self.low2high_order]  # (64, 6, 16) --> (64, 6, 16)
        DCT_blocks = DCT_blocks[:, :, :self.low_freqs]  # (64, 6, 16) --> (64, 6, low_freq_coe)

        # numpy to torch
        DCT_blocks = torch.from_numpy(DCT_blocks).reshape(self.tokens, -1)  # (64, 6*low_freq_coe)
        DCT_blocks = DCT_blocks.float()  # float64 --> float32

        return DCT_blocks
