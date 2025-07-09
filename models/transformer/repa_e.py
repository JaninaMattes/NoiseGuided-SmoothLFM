# Code based on: https://github.com/End2End-Diffusion/REPA-E/blob/main/models/sit.py
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dit_context import DiT



def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class REPA_E(DiT):
    def __init__(
        self,
        *args,
        in_channels=4,
        hidden_size=1152,
        z_dims=[768],
        encoder_depth=8,
        projector_dim=2048,
        bn_momentum=0.1,
        **kwargs
    ):
        super().__init__(*args, hidden_size=hidden_size, **kwargs)
        
        self.encoder_depth = encoder_depth
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])

        # Note that we disable affine parameters in the batch norm layer, to avoid affine hacking diffusion loss
        self.bn = torch.nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True
        )
        self.bn.reset_running_stats()
        self.initialize_weights()

    def forward(self, x, t, y=None, context=None, return_z=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        z: (N, L, C') tensor of external visual features
        loss_kwargs: dictionary of loss function arguments, should contain: `weighting`, `path_type`, `prediction`, `align_only`
        """
        x = self.x_embedder(x)                                      # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                                      # (N, D)
        
        if self.y_embedder is not None:
            # Add a null class label for unconditional generation
            if y is None:
                unconditional_idx = self.num_classes - 1
                y = torch.full((x.size(0),), unconditional_idx, dtype=torch.long, device=x.device)
            if y.ndim > 1:
                y = y.squeeze(1)

            y = self.y_embedder(y, self.training)                   # (N, D)            
            c = t + y                                               # (N, D)
        else:
            c = t
        
        if self.context_embedder is not None:
            if context is None:
                print(f"Context is None, using null vector as context")
                context_emb = torch.zeros((x.size(0), 1, x.size(2)), dtype=x.dtype, device=x.device)
            else:
                context_emb = self.context_embedder(context).unsqueeze(1)  # (N, 1, D)

            x = torch.cat([x, context_emb], dim=1)  # (N, T+1, D)
                 
        x = x + self.pos_embed                                          # (N, T+1, D)
        N, T, D = x.shape
        
        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                zs_tilde = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                
        if self.context_embedder is not None:
            x = x[:, :-1]                                               # remove context token
        
        x = self.final_layer(x, c)                                      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                                          # (N, out_channels, H, W)
        if self.learn_sigma and not self.return_sigma:                  # LEGACY
            x, _ = x.chunk(2, dim=1)

        if return_z:
            return x, zs_tilde
        return x


    def init_bn(self, latents_scale, latents_bias):
        # latents_scale = 1 / sqrt(variance); latents_bias = mean
        self.bn.running_mean = latents_bias
        self.bn.running_var = (1. / latents_scale).pow(2)

    def extract_latents_stats(self):
        # rsqrt is the reciprocal of the square root
        latent_stats = dict(
            latents_scale=self.bn.running_var.rsqrt(),
            latents_bias=self.bn.running_mean,
        )
        return latent_stats
    

def REPA_E_XL_2(**kwargs):
    return REPA_E(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def REPA_E_L_2(**kwargs):
    return REPA_E(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def REPA_E_B_2(**kwargs):
    return REPA_E(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


DiT_models = {
    'DiT-XL/2': REPA_E_XL_2,
    'DiT-L/2':  REPA_E_L_2,
    'DiT-B/2':  REPA_E_B_2,
}


if __name__ == "__main__":
    # Test the REPA_E model
    # This is a simple test to check if the model can be instantiated and run a forward pass
    torch.manual_seed(42)  # for reproducibility
    batch_size = 16
    img_size = 32
    num_actual_classes = 1000
    context_s = 1024

    ipt = torch.randn(batch_size, 4, img_size, img_size)
    t = torch.randint(0, 1000, (batch_size,))
    context_data = torch.randn(batch_size, context_s)
    
    print("--- Test 0: REPA_E ---")
    net_test0 = DiT_models['DiT-B/2'](
        num_classes=num_actual_classes,
        cat_context=True,
        input_size=img_size
    )
    net_test0.eval()
    print(net_test0)
    
    
    print("\n--- Test 1: Class-Unconditional (via y=None for null token) & Context-Conditional ---")
    
    net_test1 = DiT_models['DiT-B/2'](
        num_classes=num_actual_classes,
        cat_context=True,
        input_size=img_size
    )
    net_test1.eval()
    out_test1 = net_test1(ipt, t, y=None, context=context_data)
    
    try:
        out_test1 = net_test1(ipt, t, y=None, context=context_data)
        print("Test 1 Output:")
        print(f"{'Params':<10}: {sum([p.numel() for p in net_test1.parameters() if p.requires_grad]):,}")
        print(f"{'Input':<10}: {ipt.shape}")
        print(f"{'Output':<10}: {out_test1.shape}")
    except Exception as e:
        print(f"ERROR in Test 1: {e}")
    print("-" * 30)
    
    
    print("\n--- Test 1: Class-Unconditional (via y=None for null token, return z) & Context-Conditional ---")
    net_test2 = DiT_models['DiT-B/2'](
        num_classes=num_actual_classes,
        cat_context=True,
        input_size=img_size
    )
    
    net_test2.eval()
    try:
        out, z_repa = net_test2(ipt, t, y=None, context=context_data, return_z=True)
        print("Test 1 Output:")
        print(f"{'Params':<10}: {sum([p.numel() for p in net_test2.parameters() if p.requires_grad]):,}")
        print(f"{'Input':<10}: {ipt.shape}")

    except Exception as e:
        print(f"ERROR in Test 2: {e}")
    print("-" * 30)
    
    
    
    print("\n--- Test 2: Test REPA-E loss ---")
    net_test2 = DiT_models['DiT-B/2'](
        num_classes=num_actual_classes,
        cat_context=True,
        input_size=img_size
    )
    net_test2.eval()
    
    try:
        out, z_repa = net_test2(ipt, t, y=None, context=context_data, return_z=True)
        print("Test 2 Output:")
        print(f"{'Params':<10}: {sum([p.numel() for p in net_test2.parameters() if p.requires_grad]):,}")
        print(f"{'Input':<10}: {ipt.shape}")
        # Simulate z_sem for testing
        z_sem = [torch.randn(batch_size, 1, 768) for _ in range(len(net_test2.projectors))]
        # Compute the loss
        loss = net_test2.repa_loss(z_sem, z_repa)
        print(f"REPA-E Loss: {loss.item()}")
    except Exception as e:
        print(f"ERROR in Test 3: {e}")
    print("-" * 30)