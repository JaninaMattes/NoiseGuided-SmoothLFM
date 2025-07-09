import os 
import torch

from jutils import freeze
from ldm.frameworks.autoencoder.ae_model import AE

class AutoencoderLDMWrapper(AE):
    """Wrapper to automatically load an LDM-based Autoencoder model."""
    def __init__(self, ckpt_path: str, device='cpu'):
        # Check if the checkpoint exists
        assert os.path.exists(ckpt_path), f'[AEWrapper] Checkpoint {ckpt_path} not found!'
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        ae_cfg = checkpoint["hyper_parameters"]["ae_cfg"]
        encoder_cfg = ae_cfg["params"]["encoder_cfg"]
        decoder_cfg = ae_cfg["params"]["decoder_cfg"]
        
        # Initialize parent class (AE)
        super().__init__(encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg)
        freeze(self) # Freeze the model
        
        # Load the checkpoint
        self.ckpt_path = ckpt_path
        self.load_checkpoint(checkpoint)
        print(f'[AEWrapper] Loading checkpoint from {ckpt_path}.')
        

    def load_checkpoint(self, checkpoint: dict):
        """Load the checkpoint for the AE model."""
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