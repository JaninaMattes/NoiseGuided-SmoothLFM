# Architecture and naming

How the files map to the components of the framework, and what each name means.

## Pipeline

| Stage | Role | Config group | Thesis choice |
|---|---|---|---|
| 1 (frozen) | Image ↔ latent autoencoder, 256²×3 → 32²×4 | `first_stage/` | `sd_ae` (SD KL-autoencoder) |
| — (frozen) | Forward-time ODE that produces the guidance-free noise codes $x_t$ | `flow_encoder/` | `sit_xl_2` (class-conditional SiT-XL/2, ω = 1) |
| 2a (trained) | β-VAE self-guidance encoder $x_{t_\text{source}} \to z_t \to x_{t_\text{target}}$ | `model/` in `train_beta_vae.yaml` | `beta_vae_skip_vit_b2` |
| 2b (trained) | Self-guided velocity decoder $v_\theta(x_t, t, y, z_t)$ | `model/` in `train_flow_decoder.yaml` | `flow_decoder_sit_xl2` |
| — (frozen) | Trained β-VAE that provides $z_t$ to the decoder | `guidance_encoder/` | `beta_vae` (`paths.bvae_ckpt`) |

Data: `build_dataset.py` writes the guidance-free HDF5 dataset (`data/`, `subset/`), and both trainings read it.

## Config names

Model configs follow `<role>_<backbone>_<size><patch>`:

- `beta_vae_skip_vit_b2` is the β-VAE with a Skip-ViT encoder/decoder, base size, patch size 2. This is the thesis model.
- `beta_vae_skip_vit_s2` is a small debug variant (7 + 7 blocks, no bottleneck layer, $z_t \in \mathbb{R}^{256}$).
- `flow_decoder_sit_xl2` is the flow decoder with a SiT-XL/2 backbone. This is the thesis model.
- `flow_decoder_sit_s2` is a small debug variant whose context size matches `beta_vae_skip_vit_s2`.

Stand-ins for CPU tests carry their purpose in the name: `first_stage/pooling_stub` and `flow_encoder/dit_s2_random`
(see `ldm/testing.py`).

## Modules

| File | What it is |
|---|---|
| `ldm/frameworks/beta_vae/bvae_model.py` | `BetaVAEModel`: encoder → $(\mu, \log\sigma^2)$ → reparameterised $z_t$ → decoder. Loss `H` (Higgins β-VAE) or `B` (Burgess, capacity-annealed). |
| `ldm/frameworks/beta_vae/bvae_model_dct_conditioned.py` | Experimental variant whose decoder also receives DCT high-frequency features of the target. Not used by any config. |
| `ldm/models/autoencoder/{encoder,decoder}/skip_vit.py` | Skip-ViT: ViT with long skip connections between shallow and deep blocks. Options listed below. |
| `…/vanilla_vit.py` | Plain ViT without long skips (ablation baseline). |
| `…/mlp_res.py` | MLP and residual-MLP encoder/decoder (baseline). |
| `…/skip_vit_dct_conditioned.py` | Skip-ViT for the DCT-conditioned β-VAE above. |
| `ldm/frameworks/flow/fm.py` | `FlowModel`: flow matching around a velocity network (linear path, ODE sampling, classifier-free guidance). |
| `ldm/models/transformer/dit.py` | DiT/SiT backbone, class-conditional through AdaLN-Zero. |
| `ldm/models/transformer/dit_context.py` | DiT with a context input. $z_t$ is embedded by a two-layer MLP and appended to the token sequence (`cat_context`), with a learned null token for context dropout. |
| `ldm/models/wrapper/dit.py` | Loads the pre-trained SiT-XL/2 weights into the DiT backbone (frozen flow encoder). |
| `ldm/models/wrapper/dit_context.py` | Context DiT initialised from SiT-XL/2 (shared weights loaded, new context layers trained). |
| `ldm/models/wrapper/pretrained_beta_vae.py` | Frozen β-VAE rebuilt from a Stage-2a checkpoint (strict weight loading). |
| `ldm/trainer_beta_vae.py`, `ldm/trainer_flow_decoder.py` | Lightning modules for Stage 2a and 2b. |
| `ldm/legacy.py` | Old → new module names, so checkpoints saved before the renames still load. |

Skip-ViT options (`model.params.{encoder,decoder}_cfg.params`):

| Option | Values | Meaning |
|---|---|---|
| `skip_layer_type` | `default` (concat + linear), `self_attn`, `cross_attn` (thesis) | How skip features are fused into the deeper block |
| `skip_indices` | e.g. `[3, 6, 9, 11]` | Blocks that send/receive a long skip |
| `use_bottleneck_layer` | `True` (thesis) / `False` | Project the class token to $2 \times D_\text{btn}$ moments, or split the class token itself |
| `latent_dim` | `1024` (thesis) | Bottleneck size $D_\text{btn}$ of $z_t$ |

## Former names

The suffixes `_t1i` and `_t2i` did **not** mean text-to-image. They marked successive revisions of the same
ViT encoder/decoder (the revisions differ in the optional bottleneck layer and in the attention class). The revision
used for the thesis checkpoints (`_t2i`) is now the only one, under the plain name. `ldm/legacy.py` maps every
old name, e.g.

| Before | Now |
|---|---|
| `ldm.models.autoencoder.encoder.skip_vit_t2i.SkipViTEncoder` | `ldm.models.autoencoder.encoder.skip_vit.SkipViTEncoder` |
| `ldm.models.wrapper.beta_vae-t2i.BVaeLDMWrapper` | `ldm.models.wrapper.pretrained_beta_vae.PretrainedBetaVAE` |
| `ldm.trainer_bvae_ti2.TrainerModuleLatentBetaVae` | `ldm.trainer_beta_vae.TrainerModuleLatentBetaVae` |
| `ldm.trainer_rf_vae.TrainerModuleLatentFlow` | `ldm.trainer_flow_decoder.TrainerModuleLatentFlow` |
| `*_dctc.py` | `*_dct_conditioned.py` |
| `configs/bvae_config.yaml`, `configs/rf_config.yaml` | `configs/train_beta_vae.yaml`, `configs/train_flow_decoder.yaml` |
