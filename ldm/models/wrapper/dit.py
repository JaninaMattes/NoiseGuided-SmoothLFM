import os
import sys

import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url

from jutils import freeze

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from ldm.models.transformer.dit import DiT_models


""" DiT Wrapper """

class DiTLDMWrapper(nn.Module):
    def __init__(self, model_type, input_size, num_classes, learn_sigma=True, legacy_attn=True, ckpt_path=None, **kwargs):
        super().__init__()
        if model_type not in DiT_models:
            raise ValueError(f"Model type {model_type} not found in DiT_models. Available options are: {list(DiT_models.keys())}")
        
        self.model = DiT_models[model_type](
            input_size=input_size, num_classes=num_classes, learn_sigma=learn_sigma, **kwargs)
        freeze(self.model)

        # fetch ckpt 
        if not os.path.isfile(ckpt_path):
            os.makedirs('checkpoints', exist_ok=True)
            web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
            download_url(web_path, 'checkpoints', filename="SiT-XL-2-256x256.pt")

        # Load the checkpoint
        state_dict = torch.load(ckpt_path, weights_only=True)
        print(f'[DiTWrapper] Loading checkpoint from {ckpt_path}.')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def forward(self, x, t, **cond_kwargs):
        return self.model(x, t, **cond_kwargs)


if __name__ == "__main__":
    # Test DiTLDMWrapper
    model_type = "DiT-XL/2"
    input_size = 32
    num_classes = 1000
    learn_sigma = True
    legacy_attn = True
    ckpt_path = "checkpoints/SiT-XL-2-256x256.pt"
    
    model = DiTLDMWrapper(model_type, input_size, num_classes, learn_sigma=learn_sigma, legacy_attn=legacy_attn, ckpt_path=ckpt_path)
    print(model)

    # Test forward pass with bs=16 and y
    dev = next(model.parameters()).device
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5]*16).to(dev)
    y = torch.randint(0, 1000, (16,)).to(dev)

    with torch.no_grad():
        output = model(ipt, t, y=y)
    print(f"Output shape: {output.shape}")