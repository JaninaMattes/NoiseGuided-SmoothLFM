# Adapted from code
# - https://github.com/CompVis/fm-boosting/blob/main/fmboost/flow.py
import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)


from ldm.models.nn.layers.attention import SpatialSelfAttention
from ldm.models.nn.layers.helpers import zero_module
from ldm.models.nn.layers.pooling import Pool2d, UnPool2d
from ldm.models.nn.layers.residual import TimestepBlock
from ldm.models.nn.layers.residual import ResBlock
from ldm.models.nn.layers.resize import Downsample, Upsample
from ldm.models.nn.layers.t_emb import timestep_embedding


""" JIT Compilation """
COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    


""" QKV Attention Layers """  
        
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.embed_dimropout = dropout
        self.efficient_attn = efficient_attn
        if self.efficient_attn:
            try:
                _ = nn.functional.scaled_dot_product_attention
            except AttributeError:
                print("Please update PyTorch to 2.0 or higher to use efficient attention.")
                self.efficient_attn = False

        if COMPILE: self.forward = compile_fn(self.forward)
        

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:
            res = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.embed_dimropout)
        else:
            ch = q.shape[-1]
            scale = 1. / math.sqrt(ch)
            dot = torch.einsum('...td, ...kd -> ...tk', q, k) * scale
            weight = torch.softmax(dot, dim=-1)
            if self.embed_dimropout > 0.0:
                weight = torch.dropout(weight, p=self.embed_dimropout, train=self.training)
            res = torch.einsum('...dt, ...tv -> ...dv', weight, v)
        return res



class LinearQKVAttention(nn.Module):
    """
    A module which performs linear QKV attention.
    (https://arxiv.org/abs/1812.01243)
    """
    def __init__(self, l2_norm_v: bool = False):
        super().__init__()
        self.l2_norm_v = l2_norm_v

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        ch = q.shape[-1]
        scale = 1. / math.sqrt(ch)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * scale
        if self.l2_norm_v:
            v = torch.nn.functional.normalize(v, dim=-1)
        context = torch.einsum('...nd, ...ne -> ...de', k, v)
        res = torch.einsum('...nd, ...de -> ...ne', q, context)
        return res



class SpatialSelfAttention(nn.Module):
    """
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64,
                 use_linear: bool = False, use_efficient_attn: bool = True):
        super().__init__()
        self.embed_dimim = dim
        self.num_headseads = heads
        self.embed_dimim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.embed_dimim, 1))

        if COMPILE: self.forward = compile_fn(self.forward)
        

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (b, c, *spatial), where spatial can be (f, h, w) or (h, w).
        Returns:
            x: Tensor after attention, MHSA(x) + residual.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)                                     # (b, c, f * h * w)
        qkv = self.qkv(self.norm(x))                                # (b, 3 * c * nh, f * h * w)
        qkv = qkv.reshape(b, self.num_headseads, qkv.shape[-1], -1)         # (b, nh, f * h * w, 3 * c)
        q, k, v = qkv.chunk(3, dim=-1)                              # (b, nh, f * h * w, c) each
        h = self.attention(q, k, v)                                 # (b, nh, f * h * w, c)
        h = h.reshape(b, self.inner_dim, -1)                        # (b, nh * c, f * h * w)
        h = self.proj_out(h)                                        # (b, c, f * h * w)
        return (x + h).reshape(b, c, *spatial)


""" Efficient UNet with Context conditioning """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

""" Context Embedder """

class ContextEmbedder(nn.Module):
    """ Convert context vector (B, context_size) → (B, embed_dim) efficiently. """
    def __init__(self, context_embedding_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        if COMPILE: self.forward = compile_fn(self.forward)
        
    def forward(self, x):
        return self.mlp(x)


""" Efficient UNet with Context Conditioning """

class EfficientUNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            img_size,
            model_channels,
            num_res_blocks,
            context_dim,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            conv_resample=True,
            dim_head=64,
            num_heads=4,
            use_linear_attn=True,
            use_scale_shift_norm=True,
            pool_factor=-1,
            context_size=1024,                      # Context vector dimensionality
            cat_context=False,                      # Concatenate context to image tokens
            compile=False,
    ):
        """
        2D UNet model with attention. It includes down- and up-
        sampling blocks to train an end-to-end high resolution
        diffusion model.

        Args:
            in_channels: channels in the input Tensor.
            model_channels: base channel count for the model.
            out_channels: channels in the output Tensor.
            num_res_blocks: number of residual blocks per downsample.
            attention_resolutions: a collection of downsample rates at which
                attention will take place. Maybe a set, list, or tuple.
                For example, if this contains 4, then at 4x downsampling, attention
                will be used.
            dropout: the dropout probability.
            channel_mult: channel multiplier for each level of the UNet.
            conv_resample: if True, use learned convolutions for upsampling and
                downsampling.
            num_heads: the number of attention heads in each attention layer.
            dim_head: the dimension of each attention head.
            use_linear_attn: If true, applies linear attention in the encoder/decoder.
            use_scale_shift_norm: If True, use ScaleShiftNorm instead of LayerNorm.
            pool_factor: Down-sampling factor for spatial dimensions (w. conv layer)
        """
        super().__init__()
        global COMPILE
        COMPILE = compile
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.pool_factor = pool_factor
        self.cat_context = cat_context

        """ Time Embedder """
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Context embedding
        if cat_context:
            hidden_size = model_channels
            self.context_embedder = ContextEmbedder(context_dim, hidden_size)
            print(f"[UNet] Context-conditional ({context_size} dim)")
        else:
            self.context_embedder = None
            print("[UNet] Unconditional")
        
        """ Pooling Layer """
        # if ds factors > 1, it will down-sample the input, otherwise its just identity
        if pool_factor > 1:
            self.pool = Pool2d(in_channels, model_channels, pool_factor=pool_factor)
            starting_channels = model_channels
        else:
            self.pool = nn.Identity()
            starting_channels = in_channels

        """ Encoder and Decoder """
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(starting_channels + (in_channels if cat_context else 0), model_channels, 3, padding=1)
            )
        ])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            dim=ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialSelfAttention(
                ch,
                heads=num_heads,
                dim_head=dim_head,
                use_linear=False,
                use_efficient_attn=True,
            ),
            ResBlock(
                ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # check whether we first down-sampled the input
        if pool_factor > 1:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, 3, padding=1)
            )
            self.un_pool = UnPool2d(model_channels, out_channels, pool_factor=pool_factor)
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                zero_module(
                    nn.Conv2d(model_channels, out_channels, 3, padding=1))
            )
            self.un_pool = nn.Identity()



    def forward(self, x, t, context=None, y=None):
        """
        Apply the model to an input batch.

        Args:
            x: an [N x C x ...] Tensor of inputs.
            t: a 1-D batch of timesteps.
            context: an optional [N x C x ...] Tensor of context that gets
                concatenated to the inputs.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        # condition on semantic context
        if self.context_embedder is not None and context is not None:
            context_emb = self.context_embedder(context).unsqueeze(1)   # (N, D) -> (N, 1, D)
            x = torch.cat([x, context_emb], dim=1)                      # (N, T+1, D) append context to img tokens

        x = self.pool(x)

        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)

        h = self.out(h)
        h = self.un_pool(h)

        return h



if __name__ == "__main__":
    
    """ Test context-free UNet """
    unet = EfficientContextUNet(
        img_size=32,
        in_channels=4,
        model_channels=32,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4],
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dim_head=64,
        num_heads=4,
        use_linear_attn=True,
        use_scale_shift_norm=True,
        pool_factor=1,
        context_dim=0,
        cat_context=False,
    )
    print(f"Params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

    ipt = torch.randn((16, 4, 32, 32))
    cont = torch.randn((16, 1024))
    t_ = torch.rand((16,))
    out = unet(ipt, t_)
    print("Input:", ipt.shape)                      # (bs, c, h, w)
    print("Output:", out.shape)                     # (bs, c, h, w)
    
    
    
    
    """ Test context-conditioned UNet """
    unet = EfficientContextUNet(
        img_size=32,
        in_channels=4,
        model_channels=32,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4],
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dim_head=64,
        num_heads=4,
        use_linear_attn=True,
        use_scale_shift_norm=True,
        pool_factor=1,
        context_dim=1024,
        cat_context=True,
    )
    print(f"Params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    ipt = torch.randn((16, 4, 32, 32))
    cont = torch.randn((16, 1024))
    t_ = torch.rand((16,))
    out = unet(ipt, t_, context=cont)
    print("Input:", ipt.shape)                      # (bs, c, h, w)
    print("Output:", out.shape)                     # (bs, c, h, w)
    print("Context:", cont.shape)                   # (bs, c)