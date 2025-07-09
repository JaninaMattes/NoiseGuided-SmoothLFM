# Code is adapted from
# - https://github.com/baofff/U-ViT/blob/main/libs/uvit.py#L138
from functools import partial
import torch
import torch.nn as nn
import math
import einops
import torch.utils.checkpoint

from ldm.models.nn.layers.mlp import Mlp
from ldm.models.nn.init.weight_init import trunc_normal_
from ldm.models.nn.layers.patch_embed import PatchEmbed
from ldm.models.nn.layers.pooling import Pool2d



""" Compile function """

COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=True, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f
    
    
    
"""X formers attention mode"""

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'[UVit] Attention mode is {ATTENTION_MODE}')



""" Utility functions """

def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x



""" Attention Layer """

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


""" Block Layer """


class Block(nn.Module):

    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, 
                 compile=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

        if COMPILE: self.forward = compile_fn(self.forward)
        

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)


    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



""" Linear Projection Layer """


class ContextEmbedder(nn.Module):
    """ Convert context vector (B, 1024) → (B, hidden_size) efficiently. """
    def __init__(self, context_dim, hidden_size, act_layer=nn.SiLU, use_act=False):
        super().__init__()
        self.proj = nn.Linear(context_dim, hidden_size)                 # Directly map to hidden_size
        self.act = act_layer() if use_act else nn.Identity()            # Optional activation
        
        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x):
        x = self.proj(x)                                                # Shape: (B, hidden_size)
        x = self.act(x)
        return x


""" U-Vit with Context Conditioning """

class ContextUViT(nn.Module):
    def __init__(
        self, 
        img_size=32, 
        in_channels=4,
        patch_size=1,  
        hidden_size=768, 
        context_dim=1024,
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.,
        qkv_bias=False, 
        qk_scale=None, 
        norm_layer=nn.LayerNorm, 
        mlp_time_embed=False, 
        num_classes=-1,
        use_checkpoint=False, 
        conv=True, 
        skip=True, 
        cat_context=False,
        pool_factor=-1,
        compile=False
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        
        self.num_features = self.hidden_size = hidden_size  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pool_factor = pool_factor
        self.cat_context = cat_context

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        ) if mlp_time_embed else nn.Identity()

        # Context embedding
        if self.cat_context:
            self.context_emb = ContextEmbedder(context_dim, hidden_size)
        
        # Label embedding
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, hidden_size)
        self.extras = 1 + int(num_classes > 0) + int(cat_context)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, hidden_size))
            
        self.in_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        # Bottleneck block
        self.mid_block = Block(
                dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(hidden_size)
        self.patch_dim = patch_size ** 2 * in_channels
        self.decoder_pred = nn.Linear(hidden_size, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1) if conv else nn.Identity()

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
        
        
    def forward(self, x, t, context=None, y=None):
        """
        Forward pass of ViT. Apply model to an input batch
        
        Args:
        x: an [N x C x H x W] tensor of inputs
        timesteps: a 1-D tensor of N indices, one per batch element
        context: an [N x context_dim] tensor of context information for conditioning of the model
        y: an [N] tensor of labels optionally used for class conditioning (default: None)
        """
        x = self.patch_embed(x)                                                 # Shape: (B, L, hidden_size)
        B, L, D = x.shape

        # Add time token
        time_token = self.time_embed(timestep_embedding(t, self.hidden_size))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        
        # Add class token
        if y is not None:
            if y.ndim > 1:
                y = y.squeeze(1)
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
            
        # Add context token
        if self.cat_context and context is not None:
            context_emb = self.context_emb(context).unsqueeze(1)                # Shape: (B, 1, hidden_size)
            x = torch.cat((context_emb, x), dim=1)                              # Shape: (B, L+1, hidden_size)
        
        # Add positional encoding
        x = x + self.pos_embed                                                  # Shape: (B, L+1, hidden_size)
        
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_channels)
        x = self.final_layer(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            