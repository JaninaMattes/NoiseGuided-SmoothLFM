import os, sys
import random
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt, rcParams


from sklearn.decomposition import PCA


import gc
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from tqdm import tqdm
import numpy as np

from jutils import denorm
from jutils import ims_to_grid
from jutils.vision import tensor2im
from jutils import exists, freeze, default
from jutils import tensor2im, ims_to_grid


# Setup project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)

from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor
from ldm.trainer_rf_vae import TrainerModuleLatentFlow
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

torch.set_float32_matmul_precision('high')




#########################################################
#                    CFG/CCFG Matrix                    #
#########################################################
@torch.no_grad()
def generate_cfg_matrix(
    fm_module,
    xt_latent,
    images,
    labels=None,
    num_samples=1,
    cfg_scales=[1.0, 2.0, 3.0],
    ccfg_scales=[1.0, 2.0, 3.0],
    num_steps=50,
    num_classes=1000,
    device=None,
    save_path=None,
    title=None,
    random_sample=False,
    gt_border=8,  # Padding on the right side of the GT image
    upscale_to=128
):
    device = device or fm_module.device

    if random_sample:
        idxs = torch.randperm(xt_latent.shape[0])[:num_samples]
    else:
        idxs = torch.arange(num_samples)

    xt_latent = xt_latent[idxs].to(device)
    images = images[idxs].to(device)
    if labels is not None:
        labels = labels[idxs].to(device)

    context = fm_module.encode_third_stage(xt_latent).to(torch.float32)
    batch_size = xt_latent.size(0)

    uc_context = torch.zeros_like(context)
    uc_label = torch.full((batch_size,), num_classes, device=device, dtype=torch.long)

    grid_rows = []
    final_width = upscale_to + gt_border

    for ccfg_scale in ccfg_scales:
        labels_for_gen = labels if ccfg_scale > 0 else None
        row_imgs = []

        with torch.no_grad():
            # denorm is okay for visualization, but not for metrics
            gt_img = denorm_tensor(images[0].unsqueeze(0))[0].detach().cpu()
            gt_img = TF.resize(gt_img, [upscale_to, upscale_to])
            gt_img = F.pad(gt_img, pad=[0, gt_border, 0, 0], mode='constant', value=0)
            row_imgs.append(gt_img.unsqueeze(0))  # Shape: [1, C, H, W]

        for cfg_scale in cfg_scales:
            z = torch.randn_like(xt_latent).to(device)
            context_cond = context if cfg_scale > 0 else None # Context condition is None if cfg_scale is 0

            sample_kwargs = {
                "num_steps": num_steps,
                "progress": False,
                "context": context_cond,
                "y": labels_for_gen,
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "uc_cond_context": uc_context,
                "uc_cond": uc_label,
            }

            with torch.no_grad():
                gen = fm_module.model.generate(x=z, **sample_kwargs)
                decoded = fm_module.decode_first_stage(gen)
                # denorm is okay for visualization, but not for metrics
                decoded = denorm_tensor(decoded).detach().cpu()
                decoded = torch.stack([TF.resize(im, [upscale_to, final_width]) for im in decoded])
                row_imgs.append(decoded[0].unsqueeze(0))  # Shape: [1, C, H, W]
        
        # Stack all images in the row
        grid_rows.append(torch.cat(row_imgs, dim=0))  # [1 + len(cfg_scales), C, H, W]

    # Stack all rows vertically
    full_grid = torch.cat(grid_rows, dim=0)  # [rows * (1+cfg), C, H, W]


    # Create the grid image
    grid_img = make_grid(full_grid, nrow=len(cfg_scales) + 1, padding=0)
    rcParams.update({'font.size': 14, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(grid_img.shape[2] / 40, grid_img.shape[1] / 40))
    ax.imshow(grid_img.permute(1, 2, 0).numpy())
    ax.axis('off')
    ax.set_title(title or "CFG (X) x CCFG (Y) Guidance Matrix", fontsize=16)

    # X-axis labels: one for GT + one for each CFG scale
    ax.set_xticks([
        i * (grid_img.shape[2] // (len(cfg_scales) + 1)) + (grid_img.shape[2] // (2 * (len(cfg_scales) + 1)))
        for i in range(len(cfg_scales) + 1)
    ])
    ax.set_xticklabels(["GT"] + [f"CFG: {s}" for s in cfg_scales], fontsize=12)

    # Y-axis labels: one for each CCFG scale
    ax.set_yticks([
        i * (grid_img.shape[1] // len(ccfg_scales)) + (grid_img.shape[1] // (2 * len(ccfg_scales)))
        for i in range(len(ccfg_scales))
    ])
    ax.set_yticklabels([f"CCFG: {s}" for s in ccfg_scales], fontsize=12)

    ax.set_xlabel("Context CFG Scale", fontsize=12)
    ax.set_ylabel("Class CFG Scale", fontsize=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        print(f"[INFO] Saved CFG/CCFG matrix to: {save_path}")

    plt.show()
    plt.close(fig)
    torch.cuda.empty_cache()
    gc.collect()
    
    
    

#########################################################
#                    Interpolation                      #
#########################################################
@torch.no_grad()
def interpolate_vectors(z1, z2, alpha_vals, mode="linear", dot_threshold=0.9995):
    """
    Interpolates between z1 and z2 using either 'linear' or 'slerp' mode.
    Returns a tensor of shape (B, D) where B = len(alpha_vals).
    """
    if mode == "linear":
        return torch.stack([
            (1 - alpha) * z1 + alpha * z2 for alpha in alpha_vals
        ])
    elif mode == "slerp":
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        dot = torch.dot(z1_norm, z2_norm).clamp(-1.0, 1.0)

        if torch.abs(dot) > dot_threshold:
            return torch.stack([
                torch.lerp(z1, z2, alpha) for alpha in alpha_vals
            ])
        else:
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            return torch.stack([
                torch.sin((1.0 - alpha) * omega) / sin_omega * z1_norm +
                torch.sin(alpha * omega) / sin_omega * z2_norm
                for alpha in alpha_vals
            ])
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")
    
    
    
@torch.no_grad()
def linear_interpolation_grid_with_anchors(
    cls1_latents, cls2_latents,
    cls1_images, cls2_images,
    cls1_label, cls2_label,
    fm_module,
    results_dir,
    num_pairs=6,
    num_interpolations=24,
    cfg_scale=2.0,
    ccfg_scale=1.0, 
    sample_kwargs=None,
    cond_keys=("context",),
    title=None,
    filename_suffix="interp_with_anchors",
    use_labels=False,
    num_classes=1000,
    gt_border=0, # No 
    device=None,
    interp_type='linear', # linear or slerp
    upscale_to=128,
):
    device = device or fm_module.device
    os.makedirs(results_dir, exist_ok=True)
    alpha_lin_space = torch.linspace(0, 1, num_interpolations)
    generated_rows = []
    final_width = upscale_to + gt_border

    with torch.no_grad():
        for i in range(num_pairs):
            
            # Step 1: Labels (Optional)
            if use_labels:
                half = num_interpolations // 2
                row_labels = torch.cat([
                    torch.full((half,), cls1_label, dtype=torch.long),
                    torch.full((num_interpolations - half,), cls2_label, dtype=torch.long)
                ])
                labels = row_labels.to(device)
                print(f"[INFO] Pair {i}: Using labels for interpolation: {cls1_label} -> {cls2_label}")
            else:
                labels = None
                print(f"[INFO] Pair {i}: Using no labels for interpolation.")

            # Step 2: Encoding & Interpolation in ß-VAE latent space
            x1 = cls1_latents[i].unsqueeze(0)  # DDIM xt - Shape: [1, D]
            x2 = cls2_latents[i].unsqueeze(0)  # DDIM xt - Shape: [1, D]
            context_z1 = fm_module.encode_third_stage(x1).to(torch.float32) # ß-VAE zt (1, 1024)
            context_z2 = fm_module.encode_third_stage(x2).to(torch.float32) # ß-VAE zt (1, 1024)
            context_z1 = context_z1.squeeze(0)  # Now (1024,)
            context_z2 = context_z2.squeeze(0)  # Now (1024,)
                        
            interpolation_context = interpolate_vectors(
                context_z1, context_z2, alpha_lin_space, mode=interp_type
            ).to(device)
        
            # Step 3: Generate images    
            uc_cond_context = torch.zeros_like(interpolation_context)
            uc_cond = (
                torch.full((context.size(0),), num_classes, device=device, dtype=torch.long)
                if labels is not None else None
            )

            B = interpolation_context.size(0)  # B: Number of interpolation steps
            C, H, W = x1.size(1), x1.size(2), x1.size(3)
            random_x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)

            sample_kwargs = sample_kwargs or {}
            sample_kwargs.setdefault("num_steps", 50)
            sample_kwargs.setdefault("progress", False)
            sample_kwargs["cfg_scale"] = cfg_scale
            sample_kwargs["ccfg_scale"] = ccfg_scale
            sample_kwargs["context"] = interpolation_context
            sample_kwargs["uc_cond_context"] = uc_cond_context
            sample_kwargs["y"] = labels
            sample_kwargs["uc_cond"] = uc_cond

            generated = fm_module.model.generate(x=random_x, **sample_kwargs)
            samples = fm_module.decode_first_stage(generated)

            # denorm the generated images
            row_images = denorm_tensor(samples).detach().cpu()
            row_images = torch.stack([TF.resize(im, [upscale_to, upscale_to]) for im in row_images])

            real_start = denorm_tensor(cls1_images[i].unsqueeze(0))[0].detach().cpu()
            real_start = TF.resize(real_start, [upscale_to, upscale_to])
            real_end = denorm_tensor(cls2_images[i].unsqueeze(0))[0].detach().cpu()
            real_end = TF.resize(real_end, [upscale_to, upscale_to])

            real_start_padded = F.pad(real_start, pad=[0, gt_border, 0, 0], mode='constant', value=0)
            real_end_padded = F.pad(real_end, pad=[gt_border, 0, 0, 0], mode='constant', value=0)

            left_pad = gt_border // 2
            right_pad = gt_border - left_pad
            row_images_padded = torch.stack([
                F.pad(im, pad=[left_pad, right_pad, 0, 0], mode='constant', value=0) for im in row_images
            ])

            full_row = torch.cat([real_start_padded.unsqueeze(0), row_images_padded, real_end_padded.unsqueeze(0)], dim=0)
            generated_rows.append(full_row)

            del interpolation_context, row_images, row_images_padded
            torch.cuda.empty_cache()
            gc.collect()

    all_rows = torch.cat(generated_rows, dim=0)
    nrow = num_interpolations + 2
    grid = torchvision.utils.make_grid(all_rows, nrow=nrow, padding=0)
    grid_np = grid.permute(1, 2, 0).numpy()

    rcParams.update({'font.size': 14, 'font.family': 'DejaVu Sans'})

    cell_width = grid_np.shape[1] / nrow
    cell_height = grid_np.shape[0] / num_pairs
    fig_width = grid_np.shape[1] / 100
    fig_height = grid_np.shape[0] / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(grid_np)
    ax.axis('off')

    # === Draw only 2 vertical lines ===
    ax.axvline(x=cell_width, color='black', linewidth=6)  # After Real A
    ax.axvline(x=(nrow - 1) * cell_width, color='black', linewidth=6)  # Before Real B

    # Set X ticks
    xtick_positions = [(i + 0.5) * cell_width for i in range(nrow)]
    xtick_labels = ['Real A'] + [f'{alpha:.2f}' for alpha in alpha_lin_space] + ['Real B']
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45)

    # Set Y ticks
    ytick_positions = [(i + 0.5) * cell_height for i in range(num_pairs)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f'Pair {i}' for i in range(num_pairs)], fontsize=10)

    # Title and save
    if title is None:
        title = f"Latent Interpolation with Anchors\n"
    plt.title(title, fontsize=14)

    output_path = os.path.join(results_dir, f'interpolation_{filename_suffix}.png')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=500)
    print(f"[INFO] Saved interpolation grid with anchors to: {output_path}")
    plt.show()
    plt.close(fig)

    del grid, grid_np, all_rows
    torch.cuda.empty_cache()
    gc.collect()





##########################################################################
#           Step 1: Find Directions with linear PCA                      #
##########################################################################
@torch.no_grad()
def traverse_latents_pca(
    module, 
    sample, 
    images,  # ← now correctly used
    pca_directions,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    w=128, 
    value_range=(-6, 6), 
    n_steps=10,
    device=None, 
    results_dir='results',
    labels=None, 
    context_dim=1024,
    num_directions_to_traverse=7,
    sample_kwargs=None,
    use_labels=False,
    upscale_to=128,
):
    os.makedirs(results_dir, exist_ok=True)
    device = device or (module.device if hasattr(module, 'device') else 'cpu')

    x_vals = torch.linspace(*value_range, n_steps, device=device)
    y_vals_indices = torch.arange(num_directions_to_traverse, device=device)

    batch_size = 1
    img_h, img_w = upscale_to, upscale_to
    img_grid = np.zeros((num_directions_to_traverse * img_h, (n_steps + 1) * img_w, 3), dtype=np.uint8)

    z = torch.randn_like(sample).to(device)
    encoded = module.third_stage.encode(sample.to(device))
    base_latent = encoded['latent_dist'].sample()[0:1]  # (1, D)

    pca_tensor = torch.from_numpy(pca_directions).float().to(device)
    uc_cond_context = torch.zeros((batch_size, context_dim), device=device)

    sample_kwargs = sample_kwargs or {}

    if use_labels and labels is not None:
        i = torch.randint(0, labels.shape[0], (1,)).item()
        label = labels[i].item()
        class_cond = torch.full((batch_size,), label, dtype=torch.long, device=device)
        num_classes = sample_kwargs.get("num_classes", 1000)
        uc_cond = torch.full((batch_size,), num_classes, dtype=torch.long, device=device)
    else:
        class_cond = None
        uc_cond = None

    for i, pca_idx in enumerate(y_vals_indices):
        direction = pca_tensor[pca_idx].unsqueeze(0)

        # Insert GT image at column 0
        gt_img = denorm_tensor(images)[0].detach().cpu()
        gt_img = TF.resize(gt_img, [img_h, img_w])
        gt_img_np = gt_img.permute(1, 2, 0).numpy()
        gt_img_np = (gt_img_np - gt_img_np.min()) / (gt_img_np.max() - gt_img_np.min() + 1e-6)
        gt_img_np = (gt_img_np * 255).astype(np.uint8)
        img_grid[i * img_h:(i + 1) * img_h, 0:img_w] = gt_img_np

        for j, x in enumerate(x_vals):
            lat_mod = base_latent + x * direction
            context = lat_mod

            sample_kwargs.update({
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "context": context,
                "uc_cond_context": uc_cond_context,
                "y": class_cond,
                "uc_cond": uc_cond,
                "num_steps": sample_kwargs.get("num_steps", 50),
                "progress": False
            })

            generated = module.model.generate(x=z, **sample_kwargs)
            samples = module.decode_first_stage(generated)
            samples = denorm_tensor(samples).detach().cpu()

            img_tensor = samples[0][:3]
            img = TF.resize(img_tensor, [img_h, img_w])
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

            img_grid[i * img_h:(i + 1) * img_h, (j + 1) * img_w:(j + 2) * img_w] = img

    fig_width = (img_w * (n_steps + 1)) / 100
    fig_height = (img_h * num_directions_to_traverse) / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(img_grid)
    ax.set_xticks(np.arange(img_w / 2, img_grid.shape[1], img_w))
    ax.set_xticklabels(["GT"] + [f"{x:.1f}" for x in x_vals.cpu().numpy()])
    ax.set_yticks(np.arange(img_h / 2, img_grid.shape[0], img_h))
    ax.set_yticklabels([f"PC {i+1}" for i in y_vals_indices.cpu().numpy()])
    ax.set_xlabel("Traversal Value")
    ax.set_ylabel("Principal Component Direction")

    # Bold vertical line after GT column
    ax.axvline(x=img_w - 1, color='black', linewidth=6)
    ax.axis("off")

    output_path = os.path.join(results_dir, f"latent_traversal_pca_{num_directions_to_traverse}_dirs.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Saved PCA traversal grid to {output_path}")
    plt.show()
    plt.close(fig)

    torch.cuda.empty_cache()
    gc.collect()




########################################
########################################
########################################
@torch.no_grad()
def traverse_latents(
    module, 
    sample, 
    images,
    cfg_scale=3.0,
    ccfg_scale=1.0,
    w=128,
    batch_size=1,
    start_dim=0, 
    end_dim=7,
    value_range=(-4.5, 4.5), 
    n_steps=10,
    context_dim=1024, 
    device=None,
    results_dir='results', 
    labels=None,
    sample_kwargs=None,
    use_labels=False,
    upscale_to=128
):
    os.makedirs(results_dir, exist_ok=True)
    device = device or module.device
    sample_kwargs = sample_kwargs or {}

    x_vals = torch.linspace(*value_range, n_steps, device=device)
    y_vals = torch.arange(start_dim, end_dim + 1, device=device)
    num_rows = len(y_vals)
    num_cols = n_steps + 1  # +1 for GT column

    img_h, img_w = upscale_to, upscale_to
    img_grid = np.zeros((num_rows * img_h, num_cols * img_w, 3), dtype=np.uint8)

    # Handle sample shape
    if sample.dim() == 4:
        idx = torch.randint(0, sample.size(0), (1,)).item()
        sample = sample[idx:idx+1]
        real_img = images[idx]
    else:
        sample = sample.unsqueeze(0)
        real_img = images

    # Prepare GT image
    real_img = real_img[:3]
    real_img = TF.resize(real_img, [img_h, img_w])
    real_img = real_img.permute(1, 2, 0).numpy()
    real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-6)
    real_img = (real_img * 255).astype(np.uint8)

    # Class conditioning
    if use_labels and labels is not None:
        label = labels[idx]
        class_cond = torch.full((batch_size,), label, dtype=torch.long, device=device)
        num_classes = sample_kwargs.get("num_classes", 1000)
        uc_cond = torch.full((batch_size,), num_classes, dtype=torch.long, device=device)
    else:
        class_cond = None
        uc_cond = None

    uc_cond_context = torch.zeros((batch_size, context_dim), device=device)
    z = torch.randn_like(sample).to(device)

    # Encode latent
    encoded = module.third_stage.encode(sample.to(device))
    base_latents = encoded['latent_dist'].sample()[0:1]

    for i, dim in enumerate(y_vals):
        # Insert GT image in column 0
        img_grid[i * img_h:(i + 1) * img_h, 0:img_w] = real_img

        for j, x in enumerate(x_vals):
            lat_mod = base_latents.clone()
            lat_mod[0, dim] += x

            context = lat_mod
            sample_kwargs.update({
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "context": context,
                "uc_cond_context": uc_cond_context,
                "y": class_cond,
                "uc_cond": uc_cond,
                "num_steps": sample_kwargs.get("num_steps", 50),
                "progress": False
            })

            generated = module.model.generate(x=z, **sample_kwargs)
            samples = module.decode_first_stage(generated.to(device))
            samples = denorm_tensor(samples).detach().cpu()

            img_tensor = samples[0][:3]
            img = TF.resize(img_tensor, [img_h, img_w])
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

            grid_x = (j + 1) * img_w  # offset by 1 for GT column
            img_grid[i * img_h:(i + 1) * img_h, grid_x:grid_x + img_w] = img

    # Plotting
    fig_width = (img_w * num_cols) / 100
    fig_height = (img_h * num_rows) / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(img_grid)

    # Bold separator after GT image
    ax.axvline(x=img_w - 1, color='black', linewidth=6)

    # Axis labels
    ax.set_xticks(np.arange(img_w / 2, img_grid.shape[1], img_w))
    ax.set_xticklabels(["GT"] + [f"{x:.1f}" for x in x_vals.cpu().numpy()])
    ax.set_yticks(np.arange(img_h / 2, img_grid.shape[0], img_h))
    ax.set_yticklabels([f"$z_{{{int(dim)}}}$" for dim in y_vals.cpu().numpy()])
    ax.set_xlabel("Latent Traversal Values")
    ax.set_ylabel("Latent Dimensions")
    ax.axis("off")

    output_path = os.path.join(results_dir, f"latent_traversal_{start_dim}_{end_dim}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Saved latent traversal grid to: {output_path}")
    plt.show()
    plt.close(fig)

    torch.cuda.empty_cache()
    gc.collect()








##########################################################################
#                   Find Directions with PCA                             #
##########################################################################

@torch.no_grad()
def find_pca_directions(module, dataloader, source_timestep=0.5, num_components=10, device=None):
    device = device or (module.device if hasattr(module, 'device') else 'cpu')
    print(f"[INFO] Collecting latents for PCA on device: {device}")

    all_latents = []
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        source_latents = batch[f'latents_{source_timestep:.2f}'].to(device, non_blocking=True)
        encoded = module.third_stage.encode(source_latents)
        latents = encoded['latent_dist'].mode()
        all_latents.append(latents.cpu().numpy())

    combined_latents = np.vstack(all_latents)
    print(f"[INFO] Collected {combined_latents.shape[0]} latent vectors of dim {combined_latents.shape[1]}.")
    
    # Sorted by vairance (highest --> lowest)
    pca = PCA(n_components=num_components)
    pca.fit(combined_latents)

    print(f"[INFO] PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"[INFO] Total explained: {np.sum(pca.explained_variance_ratio_):.2f}")

    return pca.components_, pca.explained_variance_ratio_



##########################################################################
#                   Find Directions with K-Means                         #
##########################################################################

@torch.no_grad()
def find_kmeans_directions(module, dataloader, source_timestep=0.5, n_clusters=8, n_directions=7, device=None):
    """
    Collects latent vectors, performs K-Means clustering, and computes directions
    between cluster centroids.

    Args:
        module: The main model module containing the encoder.
        dataloader: Dataloader for the dataset.
        source_timestep: The timestep of the source latents to encode.
        n_clusters (int): The number of clusters for K-Means (k).
        n_directions (int): The number of traversal directions to compute.
        device: The device to run on.

    Returns:
        np.ndarray: An array of direction vectors.
    """
    device = device or (module.device if hasattr(module, 'device') else 'cpu')
    print(f"[INFO] Collecting latents for K-Means on device: {device}")

    all_latents = []
    # Collect a sufficient number of latents for stable clustering
    num_samples_to_collect = n_clusters * 1000 
    
    with tqdm(total=num_samples_to_collect, desc="Collecting Latents", unit="sample") as pbar:
        for batch in dataloader:
            if len(all_latents) * batch[f'latents_{source_timestep:.2f}'].shape[0] >= num_samples_to_collect:
                break
            source_latents = batch[f'latents_{source_timestep:.2f}'].to(device, non_blocking=True)
            encoded = module.third_stage.encode(source_latents)
            # Use the mode for a deterministic latent representation
            latents = encoded['latent_dist'].mode()
            all_latents.append(latents.cpu().numpy())
            pbar.update(latents.shape[0])

    combined_latents = np.vstack(all_latents)
    print(f"[INFO] Collected {combined_latents.shape[0]} latent vectors of dim {combined_latents.shape[1]}.")

    # Perform K-Means clustering
    print(f"[INFO] Performing K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(combined_latents)
    centroids = kmeans.cluster_centers_
    print("[INFO] K-Means clustering complete.")

    # Compute directions between centroids. A common approach is to find vectors
    # from a reference centroid (e.g., the first one) to all others.
    if n_directions >= n_clusters:
        print(f"[WARNING] Requested {n_directions} directions, but only {n_clusters - 1} can be computed from {n_clusters} centroids. Clipping.")
        n_directions = n_clusters - 1
        
    reference_centroid = centroids[0]
    direction_vectors = []
    for i in range(1, n_clusters):
        direction = centroids[i] - reference_centroid
        direction_vectors.append(direction / np.linalg.norm(direction)) # Normalize the direction

    # Select the top n_directions
    directions_to_return = np.array(direction_vectors[:n_directions])
    print(f"[INFO] Computed {directions_to_return.shape[0]} direction vectors.")

    return directions_to_return


@torch.no_grad()
def traverse_latents_kmeans(
    module,
    sample,
    images,
    kmeans_directions,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    w=128,
    value_range=(-10, 10),
    n_steps=10,
    device=None,
    results_dir='results',
    labels=None,
    context_dim=1024,
    num_directions_to_traverse=7,
    sample_kwargs=None,
    use_labels=False,
    upscale_to=128,
):
    """
    Traverse latent space along K-Means directions with a GT image in the first column
    and a bold vertical separator after it.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = device or (module.device if hasattr(module, 'device') else 'cpu')

    x_vals = torch.linspace(*value_range, n_steps, device=device)
    y_vals_indices = torch.arange(num_directions_to_traverse, device=device)

    batch_size = 1
    img_h, img_w = upscale_to, upscale_to
    num_cols = n_steps + 1  # +1 for GT image
    img_grid = np.zeros((num_directions_to_traverse * img_h, num_cols * img_w, 3), dtype=np.uint8)

    # Select one sample and its GT image
    if sample.dim() == 4:
        idx = torch.randint(0, sample.size(0), (1,)).item()
        sample = sample[idx:idx+1]
        real_img = images[idx]
    else:
        sample = sample.unsqueeze(0)
        real_img = images

    real_img = real_img[:3]
    real_img = TF.resize(real_img, [img_h, img_w])
    real_img = real_img.permute(1, 2, 0).numpy()
    real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-6)
    real_img = (real_img * 255).astype(np.uint8)

    z = torch.randn_like(sample).to(device)
    encoded = module.third_stage.encode(sample.to(device))
    base_latent = encoded['latent_dist'].mode()[0:1]

    kmeans_tensor = torch.from_numpy(kmeans_directions).float().to(device)
    uc_cond_context = torch.zeros((batch_size, context_dim), device=device)
    sample_kwargs = sample_kwargs or {}

    if use_labels and labels is not None:
        label = labels[idx].item()
        class_cond = torch.full((batch_size,), label, dtype=torch.long, device=device)
        num_classes = sample_kwargs.get("num_classes", 1000)
        uc_cond = torch.full((batch_size,), num_classes, dtype=torch.long, device=device)
    else:
        class_cond = None
        uc_cond = None

    for i, kmeans_idx in enumerate(y_vals_indices):
        direction = kmeans_tensor[kmeans_idx].unsqueeze(0)

        # Place GT image
        img_grid[i * img_h:(i + 1) * img_h, 0:img_w] = real_img

        for j, x in enumerate(x_vals):
            lat_mod = base_latent + x * direction
            context = lat_mod

            sample_kwargs.update({
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "context": context,
                "uc_cond_context": uc_cond_context,
                "y": class_cond,
                "uc_cond": uc_cond,
            })
            sample_kwargs.setdefault("num_steps", 50)
            sample_kwargs.setdefault("progress", False)

            generated = module.model.generate(x=z, **sample_kwargs)
            samples = module.decode_first_stage(generated)
            samples = denorm_tensor(samples).detach().cpu()

            img_tensor = samples[0][:3]
            img = TF.resize(img_tensor, [img_h, img_w])
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

            grid_x = (j + 1) * img_w  # Offset by +1 for GT
            img_grid[i * img_h:(i + 1) * img_h, grid_x:grid_x + img_w] = img

    # Plot with GT separator
    fig_width = (img_w * num_cols) / 100
    fig_height = (img_h * num_directions_to_traverse) / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(img_grid)

    # Bold vertical line after GT column
    ax.axvline(x=img_w - 1, color='black', linewidth=6)

    # Axis ticks
    ax.set_xticks(np.arange(img_w / 2, img_grid.shape[1], img_w))
    ax.set_xticklabels(["GT"] + [f"{x:.1f}" for x in x_vals.cpu().numpy()])
    ax.set_yticks(np.arange(img_h / 2, img_grid.shape[0], img_h))
    ax.set_yticklabels([f"KMeans {i+1}" for i in y_vals_indices.cpu().numpy()])
    ax.set_xlabel("Traversal Value")
    ax.set_ylabel("K-Means Direction")
    ax.axis("off")

    output_path = os.path.join(results_dir, f"latent_traversal_kmeans_{num_directions_to_traverse}_dirs.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Saved K-Means traversal grid to {output_path}")
    plt.show()
    plt.close(fig)

    torch.cuda.empty_cache()
    gc.collect()



#########################################################
#          Generate a Batch of Quality Samples         #
#########################################################
import math

def generate_samples(
    fm_module,
    images,
    xt_latent,
    labels=None,
    cfg_scale=1.0,
    ccfg_scale=1.0,
    num_steps=50,
    upscale_to=128,
    nrow=4,
    num_classes=1000,
    title="Generated Samples",
    save_path=None,
    device=None,
    batch_size_limit=16  # Limit batch size for model
):
    device = device or fm_module.device
    images = images.to(device)
    xt_latent = xt_latent.to(device)

    if labels is not None:
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.squeeze(1)

    all_real = []
    all_fake = []

    num_total = xt_latent.size(0)
    num_chunks = math.ceil(num_total / batch_size_limit)

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * batch_size_limit
            end = min((i + 1) * batch_size_limit, num_total)

            x_chunk = xt_latent[start:end]
            img_chunk = images[start:end]
            lbl_chunk = labels[start:end] if labels is not None else None

            context = fm_module.encode_third_stage(x_chunk.to(device)).to(torch.float32)

            z_single = torch.randn_like(x_chunk[0]).to(device)
            z = z_single.unsqueeze(0).expand(x_chunk.size(0), *z_single.shape).contiguous()

            uc_context = torch.zeros_like(context)
            uc_label = torch.full((x_chunk.size(0),), num_classes, device=device, dtype=torch.long)

            sample_kwargs = {
                "num_steps": num_steps,
                "progress": False,
                "context": context,
                "y": lbl_chunk,
                "cfg_scale": cfg_scale,
                "ccfg_scale": ccfg_scale,
                "uc_cond_context": uc_context,
                "uc_cond": uc_label,
            }

            generated = fm_module.model.generate(x=z, **sample_kwargs)
            generated_imgs = fm_module.decode_first_stage(generated)

            fake_images = denorm_tensor(generated_imgs).detach().cpu()
            real_images = denorm_tensor(img_chunk).detach().cpu()

            if upscale_to:
                real_images = torch.stack([TF.resize(im, [upscale_to, upscale_to]) for im in real_images])
                fake_images = torch.stack([TF.resize(im, [upscale_to, upscale_to]) for im in fake_images])

            all_real.extend(real_images)
            all_fake.extend(fake_images)

            del x_chunk, img_chunk, lbl_chunk, context, z, generated_imgs
            torch.cuda.empty_cache()
            gc.collect()

    interleaved = []
    for real, fake in zip(all_real, all_fake):
        interleaved.extend([real, fake])

    grid = make_grid(torch.stack(interleaved), nrow=nrow, padding=0)

    rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(grid.shape[2] / 50, grid.shape[1] / 50))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        print(f"[INFO] Saved reconstructed grid to: {save_path}")

    plt.show()
    plt.close(fig)
    torch.cuda.empty_cache()
    gc.collect()





#########################################################
#           Collect samples from the dataset            #
#########################################################
from collections import defaultdict
from typing import List, Tuple

@torch.no_grad()
def collect_samples(
    data,
    class_labels: List[int],
    source_timestep: float,
    samples_per_class: int = 10,
    group_name='validation'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collects a fixed number of samples per class from a validation dataloader.
    Returns latents, labels, and corresponding images.
    """
    collected_latents = defaultdict(list)
    collected_images = defaultdict(list)
    
    # Dataloader
    if group_name == 'validation':
        val_loader = data.val_dataloader()
    else:
        val_loader = data.test_dataloader()
        
    latent_key = f'latents_{source_timestep:.2f}'

    for batch_idx, batch in enumerate(val_loader):
        latents = batch[latent_key].detach().cpu()
        labels = batch['label'].view(-1).detach().cpu()
        images = batch['image'].detach().cpu()

        for label in class_labels:
            needed = samples_per_class - len(collected_latents[label])
            if needed <= 0:
                continue

            class_mask = labels == label
            selected_latents = latents[class_mask]
            selected_images = images[class_mask]

            if selected_latents.size(0) > 0:
                to_take = min(needed, selected_latents.size(0))
                collected_latents[label].extend(selected_latents[:to_take])
                collected_images[label].extend(selected_images[:to_take])

        if all(len(collected_latents[label]) >= samples_per_class for label in class_labels):
            print(f"[INFO] Collected enough samples for all classes after {batch_idx + 1} batches.")
            break

    all_latents, all_labels, all_images = [], [], []
    for label in class_labels:
        latents_list = collected_latents[label]
        images_list = collected_images[label]

        if len(latents_list) < samples_per_class:
            raise ValueError(f"Not enough samples collected for class {label}: got {len(latents_list)}")

        stacked_latents = torch.stack(latents_list[:samples_per_class], dim=0)
        stacked_images = torch.stack(images_list[:samples_per_class], dim=0)
        label_tensor = torch.full((samples_per_class,), label, dtype=torch.long)

        all_latents.append(stacked_latents)
        all_labels.append(label_tensor)
        all_images.append(stacked_images)

    return (
        torch.cat(all_latents, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_images, dim=0)
    )



def get_dataloader_by_group(data_module, group: str):
    if group == "validation":
        return data_module.val_dataloader()
    elif group == "test":
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Unsupported group: {group}")


    
def run_eval(
    checkpoint,
    data_path,
    interpolation_dict,
    project_name,
    model_name,
    results_root="results",
    group="validation",
    source_timestep=0.50,
    target_timestep=1.00,
    beta=0.1,
    samples_per_class=10,
    num_interpolations=16,
    num_pairs=6,
    cfg_scale=3.0,
    ccfg_scale=1.0,
    batch_size=32,
    max_samples=32,    
    device=None,
):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    fm_module = TrainerModuleLatentFlow.load_from_checkpoint(checkpoint, map_location='cpu')
    fm_module.eval()
    freeze(fm_module.model)
    fm_module.to(device)

    # Load data
    data = HDF5DataModule(
        hdf5_file=data_path,
        batch_size=batch_size,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        num_workers=4,
        train=False,
        validation=(group == "validation"),
        test=(group == "test"),
        group_name=group
    )
    data.setup(stage="fit" if group == "validation" else "test")

    # Setup results directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)



    # Task (0) Generate a batch of quality samples
    print("[INFO] Task (0): Generating a batch of quality samples ...")
    samples_dir = base_results_dir / "generated_samples"
    samples_dir.mkdir(exist_ok=True)

    dataloader = get_dataloader_by_group(data, group)
    dataloader.shuffle = True  

    # max_batches = 3
    # for batch_idx, batch in enumerate(dataloader):
    #     if batch_idx >= max_batches:
    #         break

    #     images = batch['image']
    #     latents = batch[f'latents_{source_timestep:.2f}']
    #     labels = batch['label']

    #     # 1. No conditioning
    #     generate_samples(
    #         fm_module=fm_module,
    #         images=images,
    #         xt_latent=latents,
    #         labels=None,
    #         cfg_scale=1.0,
    #         ccfg_scale=1.0,
    #         device=device,
    #         title=f"Generated Samples CFG1.0/CCFG1.0 (Batch {batch_idx})",
    #         save_path=str(samples_dir / f"cfg1_ccfg1_grid_{batch_idx}.png")
    #     )
    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     # 2. Moderate CFG
    #     generate_samples(
    #         fm_module=fm_module,
    #         images=images,
    #         xt_latent=latents,
    #         labels=None,
    #         cfg_scale=2.0,
    #         ccfg_scale=1.0,
    #         device=device,
    #         title=f"Generated Samples CFG2.0/CCFG1.0 (Batch {batch_idx})",
    #         save_path=str(samples_dir / f"cfg2_ccfg1_grid_{batch_idx}.png")
    #     )
    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     # 3. Strong CFG
    #     generate_samples(
    #         fm_module=fm_module,
    #         images=images,
    #         xt_latent=latents,
    #         labels=None,
    #         cfg_scale=3.0,
    #         ccfg_scale=1.0,
    #         device=device,
    #         title=f"Generated Samples CFG3.0/CCFG1.0 (Batch {batch_idx})",
    #         save_path=str(samples_dir / f"cfg3_ccfg1_grid_{batch_idx}.png")
    #     )
    #     torch.cuda.empty_cache()
    #     gc.collect()


    # Collect samples for interpolation and matrix generation
    all_classes = {cls for pair in interpolation_dict.values() for cls in pair}
    latents, labels, images = collect_samples(
        data=data,
        class_labels=list(all_classes),
        source_timestep=source_timestep,
        samples_per_class=samples_per_class,
        group_name=group
    )
    

    # Perform interpolations
    print("[INFO] Task (1): Linear interpolation ... ")
    for interp_name, (cls_a, cls_b) in interpolation_dict.items():
        cls1_mask = labels == cls_a
        cls2_mask = labels == cls_b

        latents_1 = latents[cls1_mask][:num_pairs].to(device)
        latents_2 = latents[cls2_mask][:num_pairs].to(device)
        imgs_1 = images[cls1_mask][:num_pairs].to(device)
        imgs_2 = images[cls2_mask][:num_pairs].to(device)

        labels_1 = torch.full((num_pairs,), cls_a, dtype=torch.long, device=device)
        labels_2 = torch.full((num_pairs,), cls_b, dtype=torch.long, device=device)

        interp_dir = base_results_dir / interp_name
        interp_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Running interpolation: {interp_name} ({cls_a} → {cls_b})")

        linear_interpolation_grid_with_anchors(
            cls1_latents=latents_1,
            cls2_latents=latents_2,
            cls1_images=imgs_1,
            cls2_images=imgs_2,
            cls1_label=labels_1,
            cls2_label=labels_2,
            fm_module=fm_module,
            results_dir=str(interp_dir),
            num_pairs=num_pairs,
            num_interpolations=num_interpolations,
            cfg_scale=cfg_scale,
            ccfg_scale=ccfg_scale,
            use_labels=False,
            num_classes=1000,
            device=device,
            filename_suffix=f"{cls_a}_{cls_b}",
            title=f"Interpolation {interp_name}: {cls_a} → {cls_b}"
        )
    
        torch.cuda.empty_cache()
        gc.collect()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    

    # Create interpolation dictionary for CFG/CCFG matrices
    # Generate CFG/CCFG Matrix
    print("[INFO] Task (2): CFG/CCFG Matrix Generation ...")
    matrix_dir = base_results_dir / "cfg_matrix"
    matrix_dir.mkdir(exist_ok=True)

    target_classes = sorted(set(cls for pair in interpolation_dict.values() for cls in pair))
    print(f"[INFO] Generating matrices for {len(target_classes)} classes: {target_classes}")

    for cls_id in target_classes:
        cls_mask = labels == cls_id
        latent_cls = latents[cls_mask][:1].to(device)   # Take only one sample
        image_cls = images[cls_mask][:1].to(device)
        label_cls = labels[cls_mask][:1].to(device)

        class_dir = matrix_dir / f"class_{cls_id}"
        class_dir.mkdir(exist_ok=True)

        print(f"[INFO] Generating CFG/CCFG matrices for class {cls_id}")

        # Matrix 1: CFG sweep (class conditioning off)
        generate_cfg_matrix(
            fm_module=fm_module,
            xt_latent=latent_cls,
            images=image_cls,
            labels=None,
            num_samples=1,
            cfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            ccfg_scales=[0.0] * 5,
            num_steps=50,
            num_classes=1000,
            device=device,
            title=f"CFG only | class {cls_id}",
            save_path=class_dir / f"class_{cls_id}_cfg_only.png",
            upscale_to=128
        )
        torch.cuda.empty_cache()
        gc.collect()

        # Matrix 2: CCFG sweep (context conditioning off)
        generate_cfg_matrix(
            fm_module=fm_module,
            xt_latent=latent_cls,
            images=image_cls,
            labels=label_cls,
            num_samples=1,
            cfg_scales=[0.0] * 5,
            ccfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            num_steps=50,
            num_classes=1000,
            device=device,
            title=f"CCFG only | class {cls_id}",
            save_path=class_dir / f"class_{cls_id}_ccfg_only.png",
            upscale_to=128
        )
        torch.cuda.empty_cache()
        gc.collect()
        
        # Matrix 3: Joint CFG x CCFG sweep
        generate_cfg_matrix(
            fm_module=fm_module,
            xt_latent=latent_cls,
            images=image_cls,
            labels=label_cls,
            num_samples=1,
            cfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            ccfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            num_steps=50,
            num_classes=1000,
            device=device,
            title=f"CFG × CCFG | class {cls_id}",
            save_path=class_dir / f"class_{cls_id}_cfg_ccfg.png",
            upscale_to=128
        )
        torch.cuda.empty_cache()
        gc.collect()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    
    # Generate label swap matrices
    print("[INFO] Task (3): CFG/CCFG Label Swap Matrix ...")
    label_swap_dir = base_results_dir / "label_swap"
    label_swap_dir.mkdir(exist_ok=True)

    for interp_name, (cls_a, cls_b) in interpolation_dict.items():
        mask_a = labels == cls_a
        mask_b = labels == cls_b

        latent_a = latents[mask_a][:1].to(device)
        latent_b = latents[mask_b][:1].to(device)
        image_a = images[mask_a][:1].to(device)
        image_b = images[mask_b][:1].to(device)

        # Labels are swapped
        swapped_label_a = torch.full((1,), cls_b, dtype=torch.long, device=device)
        swapped_label_b = torch.full((1,), cls_a, dtype=torch.long, device=device)

        swap_ab_dir = label_swap_dir / f"class_{cls_a}_as_{cls_b}"
        swap_ba_dir = label_swap_dir / f"class_{cls_b}_as_{cls_a}"
        swap_ab_dir.mkdir(exist_ok=True)
        swap_ba_dir.mkdir(exist_ok=True)

        print(f"[INFO] Class {cls_a} as {cls_b}")
        generate_cfg_matrix(
            fm_module=fm_module,
            xt_latent=latent_a,
            images=image_a,
            labels=swapped_label_a,
            num_samples=1,
            cfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            ccfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            num_steps=50,
            num_classes=1000,
            device=device,
            title=f"Class {cls_a} as {cls_b}",
            save_path=swap_ab_dir / f"class_{cls_a}_as_{cls_b}.png",
            upscale_to=128
        )

        torch.cuda.empty_cache()
        gc.collect()
    
        print(f"[INFO] Class {cls_b} as {cls_a}")
        generate_cfg_matrix(
            fm_module=fm_module,
            xt_latent=latent_b,
            images=image_b,
            labels=swapped_label_b,
            num_samples=1,
            cfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            ccfg_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
            num_steps=50,
            num_classes=1000,
            device=device,
            title=f"Class {cls_b} as {cls_a}",
            save_path=swap_ba_dir / f"class_{cls_b}_as_{cls_a}.png",
            upscale_to=128
        )

    torch.cuda.empty_cache()
    gc.collect()

           

    print("[INFO] Task (4): PCA & K-Means Traversals ...")
    pca_dir = base_results_dir / "pca_traversals"
    kmeans_dir = base_results_dir / "kmeans_traversals"
    pca_dir.mkdir(exist_ok=True)
    kmeans_dir.mkdir(exist_ok=True)

    dataloader = data.test_dataloader() if group == "test" else data.val_dataloader()
    num_directions = 10
    n_steps = 13
    traverse_range = (-3.5, 3.5)

    # Compute PCA directions once
    print("[INFO] Computing PCA directions ...")
    pca_directions, _ = find_pca_directions(
        fm_module, 
        dataloader,
        source_timestep=source_timestep, 
        num_components=num_directions, 
        device=device
    )

    # Compute KMeans directions once
    print("[INFO] Computing K-Means directions ...")
    kmeans_directions = find_kmeans_directions(
        fm_module,
        dataloader,
        source_timestep=source_timestep,
        n_clusters=50,  # Number of clusters for K-Means
        n_directions=num_directions,
        device=device
    )

    # Iterate over defined class pairs
    for interp_name, (cls_a, cls_b) in interpolation_dict.items():
        for cls_id in [cls_a, cls_b]:
            cls_mask = labels == cls_id
            latent = latents[cls_mask][:1].to(device)
            image = images[cls_mask][:1].to(device)
            label = labels[cls_mask][:1].to(device)

            traverse_latents_pca(
                fm_module,
                sample=latent,
                images=image,
                pca_directions=pca_directions,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                w=128,
                value_range=traverse_range,
                n_steps=n_steps,
                device=device,
                results_dir=str(pca_dir / f"class_{cls_id}"),
                labels=label,
                num_directions_to_traverse=num_directions,
                upscale_to=128
            )

            torch.cuda.empty_cache()
            gc.collect()
    
            # === KMeans ===
            out_dir = kmeans_dir / f"class_{cls_id}"
            out_dir.mkdir(exist_ok=True)
            print(f"[INFO] K-Means Traversal | Class {cls_id}")

            traverse_latents_kmeans(
                fm_module,
                sample=latent,
                images=image,
                kmeans_directions=kmeans_directions,
                cfg_scale=cfg_scale,
                ccfg_scale=ccfg_scale,
                value_range=traverse_range,
                n_steps=n_steps,
                device=device,
                results_dir=str(kmeans_dir / f"class_{cls_id}"),
                labels=label,
                num_directions_to_traverse=num_directions,
                upscale_to=128
            )

    torch.cuda.empty_cache()
    gc.collect()





if __name__ == "__main__":
    
    #####################################
    # Evaluation Parameters
    #####################################
    # Model checkpoints
    source_timestep     = 0.50
    target_timestep     = 1.00
    beta                = 0.1     # Beta value for the VAE
    dataset_name        = 'imagenet256-testset-T151412'
    group               = "test"  # or "test"
    baseline            = (source_timestep == 0.50 and target_timestep == 0.50)
    batch_size          = 32
    samples_per_class   = 13
    num_pairs           = 12
    num_interpolations  = 20
    cfg_scale           = 3.0
    ccfg_scale          = 1.0
    
    #####################################
    # Model Paths for SiT-XL-2
    #####################################
    
    # beta: 0.1
    DiTSXL_Beta05x05x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_0.1b/BetaVAE-B-2/2025-06-11/29847/checkpoints/last.ckpt'   ### (Baseline)
    DiTSXL_Beta00x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.00x-1.00x_0.1b/BetaVAE-B-2/2025-06-28/30448/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA02x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_0.1b/BetaVAE-B-2/2025-06-27/30400/checkpoints/last.ckpt'   ### Done
    DITSXL_BETA05x10x_01b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_0.1b/BetaVAE-B-2/V0/2025-07-03/30683/checkpoints/last.ckpt'  
    
    # beta: 1.0
    DITSXL_Beta05x05x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_1.0b/BetaVAE-B-2/2025-06-14/29969/checkpoints/last.ckpt'    
    DITSXL_Beta02x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_1.0b/BetaVAE-B-2/2025-06-13/29903/checkpoints/last.ckpt'   
    DITSXL_Beta05x10x_1b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_1.0b/BetaVAE-B-2/2025-06-18/30121/checkpoints/last.ckpt'          

    # beta: 5.0
    DiTSXL_Beta05x05x_5b ='./logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-0.50x_5.0b/BetaVAE-B-2/2025-06-19/30139/checkpoints/last.ckpt'
    DiTSXL_Beta02x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.20x-1.00x_5.0b/BetaVAE-B-2/2025-06-16/30028/checkpoints/last.ckpt'    
    DiTSXL_Beta05x10x_5b = './logs_dir/imnet256/SiT-XL-2/context_cls_cond_w_dropout/0.50x-1.00x_5.0b/BetaVAE-B-2/2025-06-19/30136/checkpoints/last.ckpt'      
    
    
    
    #####################################
    # Dataset & Evaluation Parameters
    #####################################
    checkpoint                   = DITSXL_BETA05x10x_01b
    test_data_path               = 'dataset/processed/testset-256/imagenet256-testset-T151412.hdf5' # ./dataset/processed/testset-256/imagenet256-testset-T151633.hdf5'
    validation_data_path         = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5' # './dataset/processed/testset-256/imagenet256-testset-T190319.hdf5'
    project_name                 = "CFM_Qualitative_Eval_Baseline" if baseline else "CFM_Qualitative_Eval"
    model_name                   = f"Beta-VAE-{source_timestep:.2f}x{target_timestep:.2f}x_{beta}b_{dataset_name}"



    #####################################
    # Interpolation Class Pairs
    #####################################
    interpolation_dict = {
        "admiral_to_cabbage_butterly": [321, 324],
        "monarch_to_admiral_butterly": [323, 321],
        "macaw_to_toucan": [88, 96],
        "macaw_to_cockatoo": [88, 89],
        "cockatoo_to_lorikeet": [89, 90],
        "penguin_to_flamingo": [145, 130],
        "toucan_to_penguin": [96, 145],
        "horse_to_zebra": [339, 340],
        "camel_to_gazelle": [354, 353],
        "gazelle_to_impalla": [353, 352],
        "siamese_to_persian_cat": [284, 283],
        "red_panda_to_giant_panda": [387, 388],
        "koala_to_wombat": [105, 106],
        "pembroke_corgi_to_cardigan_corgi": [263, 264],
        "pembroke_corgi_to_cockerspaniel": [263, 219],
        "husky_to_doberman": [250, 236],
        "goldenretriever_to_chowchow": [207, 260],
        "dalmatian_to_goldenretriever": [251, 207],
        "japanese_spaniel_to_dalmatian": [152, 251],
        "fox_to_husky": [277, 250],
        "lion_to_tiger": [291, 292],
        "grey_to_white_wolf": [269, 270],
        "white_wolf_to_fox": [270, 277],
        "snow_leopard_to_leopard": [289, 288],
        "brown_to_icebear": [294, 296],
        "icebear_to_brownbear": [296, 294],
        "gibbon_to_orangutan": [368, 365],  
        "lorikeet_to_peacock": [90, 84],
        # Note: Add more class ID pairs
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)

    assert num_pairs <= samples_per_class, \
        f"num_pairs ({num_pairs}) cannot be greater than samples_per_class ({samples_per_class})"



    run_eval(
        checkpoint=checkpoint,
        data_path=test_data_path if group == "test" else validation_data_path,
        interpolation_dict=interpolation_dict,
        project_name=project_name,
        model_name=model_name,
        device=device,
        group=group,  # or "test"
        source_timestep=source_timestep,
        target_timestep=1.00, # FM timestep
        beta=beta,
        samples_per_class=samples_per_class,
        num_interpolations=num_interpolations,
        num_pairs=num_pairs,
        cfg_scale=cfg_scale,
        ccfg_scale=ccfg_scale,
        batch_size=batch_size,
    )


# CUDA_VISIBLE_DEVICES=1 python '/export/home/ra93jiz/dev/Img-IDM/ldm/evaluation/fm_qualitative_eval.py'