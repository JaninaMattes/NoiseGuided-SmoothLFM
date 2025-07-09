# Code is adapted from:
# DCTDiff:
# - https://github.com/forever208/DCTdiff/blob/DCTdiff/datasets.py
# SiT:
# - https://github.com/willisma/SiT/blob/main/models.py
#
#
# MIT License
# --------------------------------------------------------
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Callable, Optional, Tuple, Union
import cv2
import numpy as np
import math

from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.fft


from timm.layers import Format, nchw_to, to_2tuple, _assert
from timm.models.vision_transformer import Attention, Mlp

from ldm.models.dct_condition.code2dtc import  DCTEmbedder
from ldm.models.dct_condition.dct.torch_dct import idct_2d
from ldm.models.nn.layers.patch_embed import PatchEmbed
from ldm.models.transformer.sit import SiT


COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        
        if COMPILE: self.forward = compile_fn(self.forward)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        if COMPILE: self.forward = compile_fn(self.forward)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

    
    
#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    


class DCTEmbedder(nn.Module):
    """
    DCT Embedder for SiT (working in latent space).
    Uses the custom DCT and IDCT functions for embedding.
    Converts latent patches (B, C, H, W) -> (B, tokens, embed_dim)
    P = 2 * B (patch_size = 2 * block_size)
    """
    def __init__(
        self,
        img_size: int = 32, 
        patch_size: int = 2,
        tokens: int = 0,
        in_channels: int = 4,
        embed_dim: int = 768,
        norm_layer=nn.LayerNorm,
        bias: bool = True,
        norm='ortho', 
        remove_low_freq_ratio=0.1,
        high_freqs=0,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.remove_low_freq_ratio = remove_low_freq_ratio
        self.DCT_coes = high_freqs
        self.tokens = tokens
        self.norm = norm

        # Optionally normalize DCT coefficients
        self.norm_layer = norm_layer(embed_dim)
        self.proj = nn.Linear(self.DCT_coes * 6, embed_dim, bias=bias)      # Shape: (DCT_coes * 6) -> embed_dim

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches
  
    def batch_split_blocks(self, x, block_sz):
        """ Splits batch tensor into (B, num_blocks, block_sz, block_sz) """
        B, H, W = x.shape[:3]
        x = x.unfold(1, block_sz, block_sz).unfold(2, block_sz, block_sz)   # (B, H//block_sz, W//block_sz, block_sz, block_sz)
        x = x.contiguous().view(B, -1, block_sz, block_sz)
        return x                                                            # Shape: (B, num_patches, patch_sz, patch_sz)
    
    def forward(self, x):
        """
        x: (N, C, H, W) input image or latent representation
        Returns: (N, T, D) DCT tokens
        """
        B, C, H, W = x.shape
        assert H == W, "Input must be square for DCT"

        # Step 1: Split the image into non-overlapping blocks (patches)
        patches = self.batch_split_blocks(x, self.patch_size[0])        # (B, C, num_patches, patch_sz, patch_sz)
        
        # Step 2: Apply DCT-2D to each patch
        dct_patches = self.dct_2d(patches, norm=self.norm)              # (B, C, num_patches, patch_sz, patch_sz)

        # Step 3: Optionally remove low-frequency components
        if self.remove_low_freq_ratio > 0:
            low_freq_threshold = int(self.remove_low_freq_ratio * dct_patches.shape[-1])
            dct_patches[:, :, :, :low_freq_threshold] = 0               # Zero out low-frequencies
        
        # Step 4: Flatten the DCT coefficients to tokens
        dct_patches = dct_patches.flatten(2).transpose(1, 2)            # (B, num_patches, D)
        
        # Step 5: Optionally normalize the resulting tokens
        x = self.norm_layer(dct_patches)
        x = self.proj(x)                                                # (B, tokens, num_low_freq*6) --> (B, tokens, hidden_dim)        
        return x



class FreqPredictionLayer(nn.Module):
    """ 
    Linear prediction layer for DCT coefficients.
    
    Args:   
        hidden_size: int, hidden size of the model
        dct_coeff: int, number of DCT coefficients (N, T, D) -> (N, T, dct_coeff)
    """
    def __init__(self, hidden_size, dct_coeff=1):
        super().__init__()
        self.linear_layer = nn.Linear(hidden_size, dct_coeff, bias=True)
        
        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        return self.linear_layer(x)
    



class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



#################################################################################
#                                 DCT SiT Model                                 #
#################################################################################
class SiTDCT(SiT):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,                       # block size of each DCT patch
        in_channels=4,
        out_channels=None,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        remove_low_freq_ratio=0.1,
        dct_norm='ortho',                   # 'ortho' or None
        load_from_ckpt=None,
        compile=False,
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        if learn_sigma:
            self.out_channels = in_channels * 2
        else:
            self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # DCT Patch Embedding
        self.dct_embedder = DCTEmbedder(norm=dct_norm, remove_low_freq_ratio=remove_low_freq_ratio)
        
        # Token Embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            uc = "+ unconditional" if class_dropout_prob > 0 else ""
            print(f"[DiT] Class-conditional ({num_classes} classes {uc})")
        else:
            self.y_embedder = None
            
        # Will use fixed sin-cos embedding:
        num_patches = self.dct_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        
        if load_from_ckpt is not None:
            dev = next(self.parameters()).device
            print(f"[DiT] Loading weights from {load_from_ckpt}")
            self.load_state_dict(torch.load(load_from_ckpt, map_location=dev))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        cls_token, extra_tokens = bool(self.y_embedder), int(self.extras + self.tokens)  # 1 if class-conditional, 0 otherwise
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=cls_token, extra_tokens=extra_tokens)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
        
        
    def forward(self, x, t, y=None,low_freg_sample=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.dct_embedder(x)                                    # (N, T, D) where T = H * W, D = number of frequency components per token                   
        t = self.t_embedder(t)                                      # (N, D)
        
        if self.y_embedder is not None:
            # Add a null class label for unconditional generation
            if y is None:
                unconditional_idx = self.num_classes if self.y_embedder.dropout_prob > 0 else self.num_classes - 1
                y = torch.full((x.size(0),), unconditional_idx, dtype=torch.long, device=x.device)

            if y.ndim > 1:
                y = y.squeeze(1)

            y = self.y_embedder(y, self.training)                   # (N, D)            
            c = t + y                                               # (N, D)
        else:
            c = t

            if y.ndim > 1:
                y = y.squeeze(1)

            y = self.y_embedder(y, self.training)                   # (N, D)
            c = t + y                                               # (N, D)
        else:
            c = t
            
        x = x + self.pos_embed                                          # (N, T+1, D)
        
        for block in self.blocks:
            x = block(x, c)                                         # (N, T, D)
        
        x = self.final_layer(x, c)                                  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                                      # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        
        # Apply the inverse DCT (IDCT)
        x = idct_2d(x, norm=self.dct_embedder.norm)  

        # Optional: Add high- and reconstructed low-freq back 
        if low_freg_sample is not None:
            x = low_freg_sample + x
            
        return x


    #####################################
    # Following DCTDiff in RGB pace     #
    #####################################
    def RGB_to_DCT(self, img, block_sz=8, low_freqs=0, high_freqs=0, Y_bound=None, low2high_order=None):
        B, C, H, W = img.shape
        assert C == 3

        Y_bound = torch.tensor(Y_bound, device=img.device)

        # Step 1: RGB to YCbCr
        Y, Cb, Cr = self.rgb_to_ycbcr(img)

        # Step 2: Downsample Cb & Cr (factor 2)
        Cb = F.interpolate(Cb, scale_factor=0.5, mode='bilinear', align_corners=False)
        Cr = F.interpolate(Cr, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Step 3: Split all channels into blocks
        y_blocks = self.batch_split_blocks(Y, block_sz)       # (B, y_blocks, block_sz, block_sz)
        cb_blocks = self.batch_split_blocks(Cb, block_sz)     # (B, cb_blocks, block_sz, block_sz)
        cr_blocks = self.batch_split_blocks(Cr, block_sz)     # (B, cr_blocks, block_sz, block_sz)

        # Step 4: Apply 2D DCT in batch
        dct_y = self.dct_2d(y_blocks)
        dct_cb = self.dct_2d(cb_blocks)
        dct_cr = self.dct_2d(cr_blocks)

        # Step 5: Normalize Y, Cb, Cr with Y_bound
        dct_y = dct_y / Y_bound
        dct_cb = dct_cb / Y_bound
        dct_cr = dct_cr / Y_bound

        # Step 6: Flatten blocks (B, N, block_sz*block_sz)
        dct_y = dct_y.flatten(2)
        dct_cb = dct_cb.flatten(2)
        dct_cr = dct_cr.flatten(2)

        # Step 7: Apply low/high frequency masking
        dct_y = dct_y[:, :, low2high_order]
        dct_cb = dct_cb[:, :, low2high_order]
        dct_cr = dct_cr[:, :, low2high_order]

        if low_freqs > 0:
            dct_y = dct_y[:, :, :low_freqs]
            dct_cb = dct_cb[:, :, :low_freqs]
            dct_cr = dct_cr[:, :, :low_freqs]
        elif high_freqs > 0:
            dct_y = dct_y[:, :, -high_freqs:]
            dct_cb = dct_cb[:, :, -high_freqs:]
            dct_cr = dct_cr[:, :, -high_freqs:]
        else:
            raise ValueError("Set either low_freqs or high_freqs > 0")

        # Step 8: Pack tokens [Y Y Y Y Cb Cr]
        # Example: For every Cb/Cr block, 4 corresponding Y blocks are aggregated (assumes structure is known)
        tokens = []  # list of tensors to concat later
        for idx in range(dct_cb.shape[1]):
            y_start = idx * 4
            y_tok = dct_y[:, y_start:y_start+4]  # (B, 4, freq_dim)
            cb_tok = dct_cb[:, idx:idx+1]        # (B, 1, freq_dim)
            cr_tok = dct_cr[:, idx:idx+1]        # (B, 1, freq_dim)
            token = torch.cat([y_tok, cb_tok, cr_tok], dim=1)  # (B, 6, freq_dim)
            tokens.append(token)

        tokens = torch.stack(tokens, dim=1)  # (B, num_tokens, 6, freq_dim)
        B, T, C6, F = tokens.shape

        return tokens.reshape(B, T, C6 * F)  # (B, num_tokens, 6*freq_dim)


    def DCT_to_RGB_torch(self, sample, tokens=0, low_freqs=0, block_sz=0, reverse_order=None, resolution=0, Y_bound=None):
        device = sample.device if torch.is_tensor(sample) else 'cpu'
        
        # Step 1: Unpack shape
        num_y_blocks = tokens * 4
        num_cb_blocks = tokens
        cb_blocks_per_row = (resolution // block_sz) // 2
        Y_blocks_per_row = resolution // block_sz

        # Step 2: Clamp & reshape tokens
        sample = sample.clamp(-2, 2)
        sample = sample.view(tokens, 6, low_freqs)  # (tokens, 6, low_freqs)

        # Step 3: Fill up full DCT coeffs
        DCT = torch.zeros((tokens, 6, block_sz * block_sz), device=device)
        DCT[:, :, :low_freqs] = sample
        DCT = DCT[:, :, reverse_order]  # (tokens, 6, 64)

        Y_bound = torch.tensor(Y_bound, device=device)

        DCT_Y = DCT[:, :4, :] * Y_bound  # (tokens, 4, 64)
        DCT_Cb = DCT[:, 4, :] * Y_bound.squeeze(0)  # (tokens, 64)
        DCT_Cr = DCT[:, 5, :] * Y_bound.squeeze(0)  # (tokens, 64)

        # Step 4: Reshape into blocks
        DCT_Cb = DCT_Cb.view(num_cb_blocks, block_sz, block_sz)
        DCT_Cr = DCT_Cr.view(num_cb_blocks, block_sz, block_sz)

        # Reorder Y blocks (your custom zigzag-like logic)
        y_blocks = []
        for row in range(cb_blocks_per_row):
            temp = []
            for col in range(cb_blocks_per_row):
                idx = row * cb_blocks_per_row + col
                y_blocks.append(DCT_Y[idx, 0].view(block_sz, block_sz))
                y_blocks.append(DCT_Y[idx, 1].view(block_sz, block_sz))
                temp.append(DCT_Y[idx, 2].view(block_sz, block_sz))
                temp.append(DCT_Y[idx, 3].view(block_sz, block_sz))
            y_blocks.extend(temp)
        DCT_Y = torch.stack(y_blocks, dim=0)  # (num_y_blocks, block_sz, block_sz)

        # Step 5: Apply IDCT
        idct_y_blocks = idct_2d(DCT_Y.unsqueeze(0)).squeeze(0)
        idct_cb_blocks = idct_2d(DCT_Cb.unsqueeze(0)).squeeze(0)
        idct_cr_blocks = idct_2d(DCT_Cr.unsqueeze(0)).squeeze(0)

        # Step 6: Combine blocks back
        def combine_blocks(blocks, H, W, block_sz):
            rows = []
            blocks = blocks.view(-1, block_sz, block_sz)
            blocks_per_row = W // block_sz
            for i in range(0, len(blocks), blocks_per_row):
                row_blocks = blocks[i:i+blocks_per_row]
                rows.append(torch.cat(list(row_blocks), dim=1))
            return torch.cat(rows, dim=0)

        y_reconstructed = combine_blocks(idct_y_blocks, resolution, resolution, block_sz)
        cb_reconstructed = combine_blocks(idct_cb_blocks, resolution // 2, resolution // 2, block_sz)
        cr_reconstructed = combine_blocks(idct_cr_blocks, resolution // 2, resolution // 2, block_sz)

        # Step 7: Upsample Cb/Cr to Y resolution
        cb_up = F.interpolate(cb_reconstructed.unsqueeze(0).unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
        cr_up = F.interpolate(cr_reconstructed.unsqueeze(0).unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()

        # Step 8: YCbCr to RGB (all values expected in [0,255] scale)
        R = y_reconstructed + 1.402 * (cr_up - 128)
        G = y_reconstructed - 0.344136 * (cb_up - 128) - 0.714136 * (cr_up - 128)
        B = y_reconstructed + 1.772 * (cb_up - 128)

        rgb = torch.stack([R, G, B], dim=0)  # (3, H, W)
        rgb = rgb.clamp(0, 255).byte()  # convert to uint8

        return rgb.permute(1, 2, 0)  # (H, W, 3)


    def rgb_to_ycbcr(self, img):
        R, G, B = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
        return Y, Cb, Cr

    def batch_split_blocks(self, x, block_sz):
        """ Splits batch tensor into (B, num_blocks, block_sz, block_sz) """
        B, H, W = x.shape[:3]
        x = x.unfold(1, block_sz, block_sz).unfold(2, block_sz, block_sz)  # (B, H//block_sz, W//block_sz, block_sz, block_sz)
        x = x.contiguous().view(B, -1, block_sz, block_sz)
        return x

    def split_into_blocks(self, img, block_sz):
        blocks = []
        for i in range(0, img.shape[0], block_sz):
            for j in range(0, img.shape[1], block_sz):
                blocks.append(img[i:i + block_sz, j:j + block_sz])  # first row, then column
        return np.array(blocks)

    def combine_blocks(self, blocks, height, width, block_sz):
        img = np.zeros((height, width), np.float32)
        index = 0
        for i in range(0, height, block_sz):
            for j in range(0, width, block_sz):
                img[i:i + block_sz, j:j + block_sz] = blocks[index]
                index += 1
        return img

    def dct_transform(self, blocks):
        dct_blocks = []
        for block in blocks:
            dct_block = np.float32(block) - 128  # Shift to center around 0
            dct_block = cv2.dct(dct_block)
            dct_blocks.append(dct_block)
        return np.array(dct_blocks)

    def idct_transform(self, blocks):
        idct_blocks = []
        for block in blocks:
            idct_block = cv2.idct(block)
            idct_block = idct_block + 128  # Shift back
            idct_blocks.append(idct_block)
        return np.array(idct_blocks)


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}