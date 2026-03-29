import logging
import os
import sys
import torch
from torchvision.datasets.utils import download_url

from jutils import freeze
from jutils.nn import AutoencoderKL
from jutils import instantiate_from_config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from ldm.models.transformer.dit import DiT_models
from ldm.frameworks.flow.fm import FlowModel




#############################################################
#                           SiT                             # 
#############################################################

def setup_SiT(input_size=32, num_classes=1000, model_type="DiT-XL/2", device='cuda'):
    model = DiT_models[model_type](
        input_size=input_size,
        num_classes=num_classes,
        learn_sigma=True,               # we "learn sigma" but never use it in SiT/DiT
        legacy_attn=True,               # legacy mode for compatibility with older checkpoints
    ).to(device)
    freeze(model)

    ckpt_path = f"checkpoints/SiT-XL-2-256x256.pt"
    if not os.path.isfile(ckpt_path):
        os.makedirs('checkpoints', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
        download_url(web_path, 'checkpoints', filename="SiT-XL-2-256x256.pt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"SiT Params: {sum([p.numel() for p in model.parameters()]):,}")
    flow = FlowModel(model, schedule="linear").to(device)
    return flow






#############################################################
#                          SD VAE                           # 
#############################################################
def setup_AE(ckpt_path=None, embed_dim=4, device='cuda'):
    """Setup Autoencoder model."""
    if ckpt_path is None:
        ckpt_path = "checkpoints/sd_ae.ckpt"
    autoencoder = AutoencoderKL(ckpt_path=ckpt_path, embed_dim=embed_dim, device=device)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    logging.info(f"AE Params: {sum([p.numel() for p in autoencoder.parameters()]):,}")
    return autoencoder





if __name__ == "__main__":
    
    
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:2")
    ipt = torch.randn(16, 4, 32, 32).to(device)
    model_type = 'ViT'
    
    # Test SiT setup
    model = setup_SiT(device=device)
    print(model)
    
    # Test Autoencoder setup
    model = setup_AE(device=device)
    print(model)

    z = model.encode(ipt, normalize=True)
    print(f"Resulting z shape: {z.shape}")