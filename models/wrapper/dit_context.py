import os
import sys
import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url

from jutils import freeze

# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from ldm.models.transformer.dit_context import DiT


def load_partial_state_dict(model: nn.Module, state_dict: dict):
    # Unwrap nested state_dict if needed
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        print("[load_partial_state_dict] Unwrapping nested 'state_dict' from checkpoint.")
        state_dict = state_dict['state_dict']

    # Skip 'pos_embed' if shapes mismatch
    if 'pos_embed' in state_dict and 'pos_embed' in model.state_dict():
        if state_dict['pos_embed'].shape != model.state_dict()['pos_embed'].shape:
            print(f"[load_partial_state_dict] Skipping 'pos_embed' due to shape mismatch.")
            del state_dict['pos_embed']

    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    ignored_keys = [k for k in state_dict if k not in filtered_dict]
    if ignored_keys:
        print(f"[load_partial_state_dict] Ignored {len(ignored_keys)} unmatched keys. Example: {ignored_keys[:5]}")

    model_dict.update(filtered_dict)
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    print(f"[load_partial_state_dict] Missing keys: {missing_keys}")
    print(f"[load_partial_state_dict] Unexpected keys: {unexpected_keys}")


class DiTLDMWrapper(DiT):
    def __init__(
        self,
        input_size,
        num_classes,
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        learn_sigma=True,
        legacy_attn=True,
        ckpt_path="checkpoints/SiT-XL-2-256x256.pt",
        freeze_backbone=False,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            **kwargs
        )

        if ckpt_path is not None and not os.path.isfile(ckpt_path):
            os.makedirs('checkpoints', exist_ok=True)
            print(f"[DiTWrapper] Downloading pretrained weights to {ckpt_path}")
            web_path = (
                "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/"
                "SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"
            )
            download_url(web_path, 'checkpoints', filename=os.path.basename(ckpt_path))

        if ckpt_path is not None:
            print(f"[DiTWrapper] Loading partial weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            load_partial_state_dict(self, state_dict)
            print("[DiTWrapper] Weights loaded. You may call `.eval()` manually if needed.")

        if freeze_backbone:
            print("[DiTWrapper] Freezing all model parameters.")
            freeze(self)


if __name__ == "__main__":
    # Test DiTLDMWrapper
    model_type = "DiT-XL/2"
    input_size = 32
    num_classes = 1000
    depth=28
    hidden_size=1152
    patch_size=2
    num_heads=16
    learn_sigma = True
    legacy_attn = True
    ckpt_path = "checkpoints/SiT-XL-2-256x256.pt"

    model = DiTLDMWrapper(
        input_size,
        num_classes,
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
        learn_sigma=learn_sigma,
        legacy_attn=legacy_attn,
        ckpt_path=ckpt_path,
        freeze_backbone=False  # Set to True if you want to freeze layers
    )
    print(model)

    # Test forward pass
    dev = next(model.parameters()).device
    ipt = torch.randn(16, 4, 32, 32).to(dev)
    t = torch.tensor([0.5] * 16).to(dev)
    y = torch.randint(0, 1000, (16,)).to(dev)
    context = torch.randn(16, 1024).to(dev)  # Example context tensor

    with torch.no_grad():
        output = model(ipt, t, y=y, context=context)
    print(f"Output shape: {output.shape}")
