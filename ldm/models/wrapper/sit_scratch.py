import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url
from functools import partial

from jutils import freeze

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from ldm.models.context_diffusion.sit_context import SiT_models


""" SiT Wrapper for LDM """
    
class SiTLDMWrapper(nn.Module):
    def __init__(
        self, 
        model_type, 
        learn_sigma=False, 
        legacy_attn=True, 
        ckpt_path=None, 
        sit_model=SiT_models, 
        requires_grad=True, 
        **kwargs
    ):
        super().__init__()

        if model_type not in sit_model:
            raise ValueError(f"Invalid model_type '{model_type}'. Choose from {list(sit_model.keys())}")

        self.model = sit_model[model_type](learn_sigma=learn_sigma, **kwargs)

        if not requires_grad:
            freeze(self.model)
            print("[SiTWrapper] Model is frozen.")

        if ckpt_path:
            print(f"[SiTWrapper] Loading checkpoint from: {ckpt_path}")
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path):
        # Load checkpoint safely
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # Clean key names
        cleaned_state_dict = {
            k.replace("model.net.", ""): v
            for k, v in checkpoint.items()
        }

        # Debugging output
        missing_keys = set(self.model.state_dict().keys()) - set(cleaned_state_dict.keys())
        if missing_keys:
            print(f"[SiTWrapper] Missing keys in checkpoint: {missing_keys}")

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        print("[SiTWrapper] Checkpoint loaded successfully.")

    def forward(self, x, t, **cond_kwargs):
        return self.model(x, t, **cond_kwargs)



if __name__ == "__main__":
    # --- Configuration for SiTContext Wrapper ---
    model_type = "SiT-S/2" 
    input_size = 32
    in_channels = 4
    mlp_ratio = 4
    num_classes = 1000
    learn_sigma = False
    legacy_attn = True
    cat_context = True
    context_size = 1024
    compile = True  
    ckpt_path     = 'checkpoints/SiT-S-2/fromscratch_step105000_cleaned.ckpt'


    model = SiTLDMWrapper(
        model_type=model_type,
        input_size=input_size,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
        legacy_attn=legacy_attn,
        cat_context=cat_context,
        context_size=context_size,
        compile=compile,
        ckpt_path=ckpt_path
    )
    print(model)

    # Test forward pass with bs=1
    dev = next(model.parameters()).device
    ipt = torch.randn(1, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]).to(dev)
    context = torch.randn(1, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=None, context=context)
    print(f"Output shape: {output.shape}")


    # Test forward pass with bs=16
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]*16).to(dev)
    context = torch.randn(16, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=None, context=context)
    print(f"Output shape: {output.shape}")
    
    # Test forward pass with bs=16 and y
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]*16).to(dev)
    y = torch.randint(0, 1000, (16,)).to(dev)
    context = torch.randn(16, 1024).to(dev)
    with torch.no_grad():
        output = model(ipt, t, y=y, context=context)
    print(f"Output shape: {output.shape}")