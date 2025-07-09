import os 
import torch

from jutils import freeze
from ldm.frameworks.beta_vae.bvae_model import BetaVAEModel


class BVaeLDMWrapper(BetaVAEModel):
    """Wrapper to automatically load an LDM-based Beta-VAE model."""
    def __init__(self, ckpt_path: str, device='cpu'):
        # Check if the checkpoint exists
        assert os.path.exists(ckpt_path), f'[BVaeWrapper] Checkpoint {ckpt_path} not found!'
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        bvae_cfg = checkpoint["hyper_parameters"]["bae_cfg"]
        encoder_cfg = bvae_cfg["params"]["encoder_cfg"]
        decoder_cfg = bvae_cfg["params"]["decoder_cfg"]
        beta = bvae_cfg["params"]["beta"]
        kld_weight = bvae_cfg["params"]["kld_weight"]
        loss_type = bvae_cfg["params"]["loss_type"]
        
        # Initialize parent class (AE)
        super().__init__(encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg, beta=beta, kld_weight=kld_weight, loss_type=loss_type)
        freeze(self)
        
        # Load checkpoint
        self.ckpt_path = ckpt_path
        self.load_checkpoint(checkpoint)
        print(f'[BVaeWrapper] Loading checkpoint from {ckpt_path}.')

    def load_checkpoint(self, checkpoint: dict):
        """Load the checkpoint for the Beta-VAE model."""
        state_dict = checkpoint['state_dict']
        
        model_state_dict = self.state_dict()
        model_keys = set(model_state_dict.keys())

        # Filter for model keys in the state_dict
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.replace('model.', '') in model_keys}
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()
        
if __name__ == "__main__":
    # Test AutoencoderLDMWrapper
    config_path = "configs/autoencoder/ae_config.yaml"
    ae_wrapper = AutoencoderLDMWrapper(config_path)
    model = ae_wrapper.get_model()
    print(model)