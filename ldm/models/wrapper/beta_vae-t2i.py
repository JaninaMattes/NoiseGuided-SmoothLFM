import os 
import sys
import torch

from jutils import freeze


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)


from ldm.frameworks.beta_vae.bvae_model_t2i import BetaVAEModel


""" Wrapper to automatically load an LDM-based Beta-VAE model. """


class BVaeLDMWrapper(BetaVAEModel):
    def __init__(self, ckpt_path: str, device='cpu'):
        assert os.path.exists(ckpt_path), f'[BVaeWrapper] Checkpoint {ckpt_path} not found!'
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Extract model config
        bvae_cfg = checkpoint["hyper_parameters"]["vae_cfg"]
        encoder_cfg = bvae_cfg["params"]["encoder_cfg"]
        decoder_cfg = bvae_cfg["params"]["decoder_cfg"]
        beta = bvae_cfg["params"]["beta"]
        kld_weight = bvae_cfg["params"]["kld_weight"]
        loss_type = bvae_cfg["params"]["loss_type"]

        super().__init__(encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg, beta=beta, kld_weight=kld_weight, loss_type=loss_type)
        freeze(self)

        self.load_checkpoint(checkpoint)
        print(f'[BVaeWrapper] Loaded checkpoint from {ckpt_path}.')

    def load_checkpoint(self, checkpoint: dict):
        state_dict = checkpoint['state_dict']
        model_keys = set(self.state_dict().keys())

        # Filter keys for only relevant model parameters
        new_state_dict = {
            k.replace('model.', ''): v
            for k, v in state_dict.items()
            if k.replace('model.', '') in model_keys
        }
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()

        
        
if __name__ == "__main__":
    # Test AutoencoderLDMWrapper
    ckpt_path = "logs_dir/imnet256/beta-vae-skipViT-S-2/0.50x-0.50x-0.1b/2025-05-06/27129/checkpoints/last.ckpt"
    model = BVaeLDMWrapper(ckpt_path)
    print(model)