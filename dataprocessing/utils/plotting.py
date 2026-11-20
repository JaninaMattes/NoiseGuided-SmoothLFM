import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging

from sklearn.decomposition import PCA
import wandb
from trimap import TRIMAP
from umap import UMAP
from pacmap import PaCMAP

import torch
import torchvision

from jutils import denorm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from experiments.utils.data_processing import decode_to_pixel_space
from experiments.utils.utility import collect_samples, denorm_tensor, z_score_normalize

# Setup font and style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12


"""Plotting for ß-VAE"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_samples_from_prior(
        beta_vae_model, autoencoder, flow, timestep=0.0, sample_kwargs=None, class1=0, n_samples=16, nrow=8, 
        denorm=False, filename='sampled_prior.png', log_dir=None, device='cuda'
    ) -> None:
    """Plot samples from the prior distribution"""
    try: 
        with torch.no_grad():
            samples = beta_vae_model.sample_prior(n_samples, device=device)
            batch_size = samples.size(0)
            y = torch.tensor([class1] * batch_size).to(device)

            # Decode to pixel space
            decoded_samples = decode_to_pixel_space(
                samples, 
                y, 
                beta_vae_model, 
                autoencoder, 
                flow, 
                t_start=timestep, 
                sample_kwargs=sample_kwargs, 
                device=device, 
                denorm=denorm,
                use_decoder=False       # already decoded by ß-VAE
            )

        # Denormalize and clamp to [0, 255]
        samples = denorm_tensor(samples.detach().cpu())
        decoded_samples = denorm_tensor(decoded_samples.detach().cpu())

        samples_grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=0)
        decoded_grid = torchvision.utils.make_grid(decoded_samples, nrow=nrow, padding=0)

        plt.figure(figsize=(15, 10))

        # Ground truth 
        plt.subplot(1, 2, 1)
        plt.imshow(samples_grid.permute(1, 2, 0).numpy())
        plt.title(f'Latent Space Prior Distribution Samples', fontsize=10)
        plt.axis('off')

        # Reconstructed samples
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_grid.permute(1, 2, 0).numpy())
        plt.title(f'Pixel Space Reconstructed Prior Distribution Samples', fontsize=10)
        plt.axis('off')

        plt.show()

        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved samples from prior to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting samples from prior: {e}")


def plot_images_grid(
        images, n_samples=8, nrow=4, title='Sample Images', filename='sample_images.png', log_dir=None):
    try: 
        images = torch.cat(images, dim=0)
        images = images.detach().cpu()
        images = denorm_tensor(images)

        grid_img = torchvision.utils.make_grid(images[:n_samples], nrow=nrow, padding=0)
        grid_img_np = grid_img.permute(1, 2, 0).numpy()

        plt.figure(figsize=(15, 15))
        plt.imshow(grid_img_np)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.show()
        
        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved samples to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting images grid: {e}")



def plot_training_losses(
    losses: Dict[str, list],
    title: str = 'Training and Validation Losses',
    filename: str = 'training_validation_losses.png',
    log_dir: str = None,
    use_wandb: bool = False
) -> None:
    """Plot training and validation losses in 3 subplots for Total, Recon, and KLD losses."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        axes[0].set_title("Total Loss")
        axes[1].set_title("Reconstruction Loss")
        axes[2].set_title("KL-Divergence Loss")

        loss_types = ['total_loss', 'recon_loss', 'kld_loss']
        for idx, loss_type in enumerate(loss_types):
            train_loss = losses[f'train_{loss_type}']
            val_loss = losses[f'val_{loss_type}']
            # Plot training and validation losses
            axes[idx].plot(range(len(train_loss)), train_loss, label=f'Train {loss_type.capitalize()}', color='red', alpha=0.7)
            axes[idx].plot(range(len(val_loss)), val_loss, label=f'Val {loss_type.capitalize()}', color='blue', alpha=0.7)
            axes[idx].set_xlabel('Steps')
            axes[idx].set_ylabel('Loss Value')
            axes[idx].legend()

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        if log_dir:
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved loss curve to {file_path}")
        
        if use_wandb:
            wandb.log({title: [wandb.Image(plt)]})

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error plotting training losses: {e}")





def plot_losses(train_losses, test_losses, 
        title='Train and Test Losses',
        filename='training_loss_curve.png', 
        log_dir=None
    ) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

    # Plot train losses
    axs[0].plot(train_losses, label="Train")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Train Loss")

    # Plot test losses
    axs[1].plot(test_losses, label="Test", color='orange')
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].set_title("Test Loss")

    # Set title
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()

    if log_dir: 
        file_path = os.path.join(log_dir, filename)
        plt.savefig(file_path, dpi=300)
        logging.info(f"Saved loss curve to {file_path}")
    plt.close()



def plot_latent_space(
    data_loader, beta_vae_model, timestep=0.0, max_batches=1000, max_samples=10000, normalize=False, 
    normalize_latents=True, embedding_method='TriMap', title='Latent Space Visualization', 
    filename='latent_space.png', log_dir=None, device='cuda'
) -> None:
    """Plot the latent space of the model"""

    beta_vae_model.eval()
    test_loader = data_loader.val_dataloader(timestep=timestep)       # TODO: Replace with val_dataloader
    collected_samples = []
    sample_counter = 0

    try:
        # Collect latent samples and labels from the test set
        for idx, batch in enumerate(test_loader):
            if idx >= max_batches:
                logging.info(f"Collected {sample_counter} samples from latent space")
                break
            
            latent_code = batch['latent'].to(device, non_blocking=True)
            label = batch['label'].to(device, dtype=torch.long, non_blocking=True)

            with torch.no_grad():
                z, _, _ = beta_vae_model.encode(latent_code.to(device), normalize=normalize)    # Get encoded z from base latent code

            # Use z-scaling
            if normalize_latents:
                z = z_score_normalize(z)
            
            collected_samples.append((z.detach().cpu().numpy(), label.detach().cpu().numpy()))
            sample_counter += len(label)
            
            if sample_counter >= max_samples:
                logging.info(f"Collected {sample_counter} samples from latent space")
                break

        # Concatenate latent samples and labels
        z, labels = zip(*collected_samples)
        z = np.concatenate(z, axis=0)
        labels = np.concatenate(labels, axis=0)
        logging.info(f"Latent space shape: {z.shape}, Labels shape: {labels.shape}")

        # Embed to 2D space
        if z.shape[1] > 2:
            if embedding_method == 'TriMap':
                embedding = TRIMAP(verbose=False).fit_transform(z)
                gs = TRIMAP(verbose=False).global_score(z, embedding)
                logging.info(f"Global Score: {gs:.2f}")
            elif embedding_method == 'PaCMAP':
                embedding = PaCMAP().fit_transform(z)
            elif embedding_method == 'UMAP':
                embedding = UMAP().fit_transform(z)
            elif embedding_method == 'PCA':
                embedding = PCA(n_components=2).fit_transform(z)
            else:
                raise ValueError(f"Unknown embedding method: {embedding_method}")
            
            # Replace z with the 2D embedding
            z = embedding

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))

        for i, label in enumerate(unique_labels):
            label_idx = labels == label
            plt.scatter(z[label_idx, 0], z[label_idx, 1], s=10, alpha=0.7, 
                        label=label, c=[cmap(i)])

        plt.title(title, fontsize=12)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), title="Classes")
        plt.tight_layout()

        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved latent visualization to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting latent space: {e}")



def plot_interpolations(
    data_loader, beta_vae_model, autoencoder, flow, timestep=0., sample_kwargs=None, n_samples=16, nrow=8, class1=0, use_latents=True,
    normalize=False, denorm=False, filename='interpolations.png', log_dir=None, device='cuda'
) -> None:
    """Plot interpolations between two latent samples"""
    beta_vae_model.eval()
    testloader = data_loader.val_dataloader(timestep=timestep)
    samples_dict = collect_samples(
        dataloader=testloader, target_classes=[class1], num_samples=2, use_latents=use_latents, max_batches=100
    )

    base_imgs1, _ = samples_dict[class1]
    x_1, x_2 = base_imgs1[0], base_imgs1[1]

    # Check for batch size
    if x_1.dim() == 3 and x_2.dim() == 3:
        x_1 = x_1.unsqueeze(0)
        x_2 = x_2.unsqueeze(0)

    try:
        # Plot interpolations between two latents
        with torch.no_grad():
            y = torch.tensor([class1] * n_samples, dtype=torch.float64).to(device)
            z1, _, _ = beta_vae_model.encode(x_1.to(device), normalize=normalize)
            z2, _, _ = beta_vae_model.encode(x_2.to(device), normalize=normalize)

            interps = [beta_vae_model.decode(z1 + (z2 - z1) * t, denorm=denorm) for t in np.linspace(0, 1, n_samples)]
            interps = torch.cat(interps)

            # Decode to pixel space
            decoded_samples = decode_to_pixel_space(
                interps, 
                y, 
                beta_vae_model, 
                autoencoder, 
                flow, 
                t_start=timestep,
                sample_kwargs=sample_kwargs, 
                device=device, 
                denorm=denorm,
                use_decoder=False               # already decoded by ß-VAE
            )

        # Denormalize and clamp to [0, 255]
        interp_samples = denorm_tensor(interps.cpu().detach())
        decoded_samples = denorm_tensor(decoded_samples.detach().cpu())

        interps_grid = torchvision.utils.make_grid(interp_samples, nrow=nrow, padding=0)
        decoded_grid = torchvision.utils.make_grid(decoded_samples, nrow=nrow, padding=0)

        plt.figure(figsize=(10, 5))

        # Subplot for interpolated samples
        plt.subplot(1, 2, 1)
        plt.imshow(interps_grid.permute(1, 2, 0).numpy())
        plt.title(f'Interpolation of Latent Samples (Class {class1})', fontsize=10)
        plt.axis('off')

        # Subplot for decoded samples
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_grid.permute(1, 2, 0).numpy())
        plt.title(f'Decoded Interpolations (Class {class1})', fontsize=10)
        plt.axis('off')

        plt.show()

        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved interpolations to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting interpolations: {e}")



def plot_interpolation_between_classes(
        data_loader, beta_vae_model, autoencoder, flow, sample_kwargs=None, class1=0, class2=1, 
        timestep=0., n_samples=16, nrow=8, use_latents=True, normalize=False, denorm=False, 
        filename='interpolation_between_classes.png', log_dir=None, device='cuda'
    ) -> None:
    """Plot interpolations between two latent samples"""
    beta_vae_model.eval()
    testloader = data_loader.val_dataloader(timestep=timestep)
    samples_dict = collect_samples(
        dataloader=testloader, target_classes=[class1, class2], num_samples=1, use_latents=use_latents, max_batches=100
    )
    base_imgs1, _ = samples_dict[class1]
    base_imgs2, _ = samples_dict[class2]

    # Check for batch size
    if base_imgs1.dim() == 3 and base_imgs2.dim() == 3:
        base_imgs1 = base_imgs1.unsqueeze(0)
        base_imgs2 = base_imgs2.unsqueeze(0)

    try:
        # Plot interpolations between two latents
        with torch.no_grad():
            z1, _, _ = beta_vae_model.encode(base_imgs1.to(device), normalize=normalize)
            z2, _, _ = beta_vae_model.encode(base_imgs2.to(device), normalize=normalize)

            interps = [beta_vae_model.decode(z1 + (z2 - z1) * t, denorm=denorm) for t in np.linspace(0, 1, n_samples)]
            interps = torch.cat(interps)
            
            # Create class labels for interpolations
            y = torch.cat([torch.tensor([class1] * (n_samples // 2)).to(device),
                           torch.tensor([class2] * (n_samples // 2)).to(device)])

            # Decode to pixel space
            decoded_samples = decode_to_pixel_space(
                interps, 
                y, 
                beta_vae_model, 
                autoencoder, 
                flow, 
                t_start=timestep,
                sample_kwargs=sample_kwargs, 
                device=device, 
                denorm=denorm,
                use_decoder=False       # already decoded by ß-VAE
            )

        # Denormalize and clamp to [0, 255]
        interp_samples = denorm_tensor(interps.detach().cpu())
        decoded_samples = denorm_tensor(decoded_samples.detach().cpu())

        interps_grid = torchvision.utils.make_grid(interp_samples, nrow=nrow, padding=0)
        decoded_grid = torchvision.utils.make_grid(decoded_samples, nrow=nrow, padding=0)

        plt.figure(figsize=(10, 5))

        # Interpolations of latent samples
        plt.subplot(1, 2, 1)
        plt.imshow(interps_grid.permute(1, 2, 0).numpy())
        plt.title(f'Interpolations of Latents (Class {class1}, Class {class2})', fontsize=10)
        plt.axis('off')

        # Reconstruction of latent samples
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_grid.permute(1, 2, 0).numpy())
        plt.title(f'Reconstructions from Interpolations (Class {class1}, Class {class2})', fontsize=10)
        plt.axis('off')

        plt.show()

        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved interpolations to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting interpolations: {e}")




def tweak_latent_dim(
    data_loader, beta_vae_model, autoencoder, flow, sample_kwargs=None, timestep=0., class1=0, value=0.5, 
    n_samples=16, nrow=8, normalize=False, denorm=False, use_latents=True, filename='tweak_latent_dims.png', log_dir=None, device='cuda'
) -> List[torch.Tensor]:
    """Tweak a latent dimension of a specific class as base keeping the value the same"""
    beta_vae_model.eval()

    testloader = data_loader.val_dataloader(timestep=timestep)
    samples_dict = collect_samples(dataloader=testloader, target_classes=[class1],
                                   num_samples=1, use_latents=use_latents, max_batches=100)
    base_imgs, _ = samples_dict[class1]

    # Check for batch size
    if base_imgs.dim() == 3:
        base_imgs = base_imgs.unsqueeze(0)

    try:
        with torch.no_grad():
            # Encode the base images
            z, _, _ = beta_vae_model.encode(base_imgs.to(device), normalize=normalize)
            tweaked_images = []
            # Tweak the latent dimension
            for dim in range(n_samples):
                z_tweaked = z.clone()
                z_tweaked[:, dim] = value
                tweaked = beta_vae_model.decode(z_tweaked, denorm=denorm)
                tweaked_images.append(tweaked)

            tweaked_images = torch.cat(tweaked_images)
            y = torch.tensor([class1] * n_samples).to(device)

            # Decode to pixel space
            decoded_samples = decode_to_pixel_space(
                tweaked_images, 
                y, 
                beta_vae_model, 
                autoencoder, 
                flow, 
                t_start=timestep, 
                sample_kwargs=sample_kwargs, 
                device=device, 
                denorm=denorm,
                use_decoder=False           # already decoded by ß-VAE
            )
        
        # Denormalize and clamp to [0, 255]
        tweaked_images = denorm_tensor(tweaked_images.detach().cpu())
        decoded_samples = denorm_tensor(decoded_samples.detach().cpu())

        tweaked_grid = torchvision.utils.make_grid(tweaked_images, nrow=nrow, padding=0)
        decoded_grid = torchvision.utils.make_grid(decoded_samples, nrow=nrow, padding=0)

        plt.figure(figsize=(10, 5))

        # Subplot Latent Samples
        plt.subplot(1, 2, 1)
        plt.imshow(tweaked_grid.permute(1, 2, 0).numpy())
        plt.title(f'Tweaked Latents (Class {class1}, Value {value:.2f})', fontsize=10)
        plt.axis('off')

        # Subplot Decoded Samples
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_grid.permute(1, 2, 0).numpy())
        plt.title(f'Decoded Latents (Class {class1}, Value {value:.2f})', fontsize=10)
        plt.axis('off')
        plt.show()

        if log_dir: 
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved tweaked latents to {file_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error tweaking latent dimensions: { e }")



def tweak_latent_values(
    data_loader, beta_vae_model, autoencoder, flow, sample_kwargs=None, timestep=0., class1=0,
    latent_dim=0, random_values=None, n_samples=8, nrow=8, normalize=False,
    use_latents=True, filename='tweak_latent_dims.png', log_dir=None, device='cuda'
):
    """
    Tweak a latent dimension with multiple random values and visualize the results in a grid.
    
    Args:
        latent_dim (int): The index of the latent dimension to tweak.
        random_values (List[float]): A list of random values to set for the selected latent dimension.
    """
    beta_vae_model.eval()

    if random_values is None:
        random_values = torch.linspace(-3, 3, n_samples).to(device)  # Default range
    else:
        random_values = torch.tensor(random_values, dtype=torch.float32).to(device)

    # Sample base images
    testloader = data_loader.val_dataloader(timestep=timestep)
    samples_dict = collect_samples(
        dataloader=testloader, target_classes=[class1],
        num_samples=n_samples, use_latents=use_latents, max_batches=100
    )
    base_imgs, _ = samples_dict[class1]

    if base_imgs.dim() == 3:
        base_imgs = base_imgs.unsqueeze(0)  # Add batch dimension if missing

    try:
        with torch.no_grad():
            # Encode the base images
            z, _, _ = beta_vae_model.encode(base_imgs.to(device), normalize=normalize)
            tweaked_images_all, decoded_images_all = [], []

            # Tweak the latent dimension with random values
            for value in random_values:
                z_tweaked = z.clone()
                z_tweaked[:, latent_dim] = value
                
                # Decode with ß-VAE
                tweaked = beta_vae_model.decode(z_tweaked.to(device), denorm=normalize)
                tweaked_images_all.append(denorm_tensor(tweaked.detach().cpu()))

                label = torch.tensor([class1] * tweaked.shape[0]).to(device)
                decoded = decode_to_pixel_space(
                    tweaked, label, beta_vae_model, autoencoder, flow, t_start=timestep, 
                    sample_kwargs=sample_kwargs, device=device, denorm=normalize, use_decoder=False
                )
                decoded_images_all.append(denorm_tensor(decoded.detach().cpu())) 

        # Create a grid of tweaked and decoded images
        tweaked_images_grid = torchvision.utils.make_grid(torch.cat(tweaked_images_all), nrow=nrow, padding=0)
        decoded_images_grid = torchvision.utils.make_grid(torch.cat(decoded_images_all), nrow=nrow, padding=0)

        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot tweaked images
        axes[0].imshow(tweaked_images_grid.permute(1, 2, 0).numpy())
        axes[0].set_title(f'Tweaked Latent ß-VAE Representations (Dim {latent_dim})')
        axes[0].axis('off')

        # Plot decoded images
        axes[1].imshow(decoded_images_grid.permute(1, 2, 0).numpy())
        axes[1].set_title('Decoded Images to Pixel Space')
        axes[1].axis('off')

        img_width = tweaked_images_grid.shape[2] // nrow
        
        # Add latent dimension annotation
        fig.text(0.02, 0.5, f'Latent Dim: {latent_dim}', 
                 va='center', ha='center', rotation='vertical', 
                 fontsize=12, color='white')

        # Add value labels for each column
        for i, value in enumerate(random_values):
            col = i % nrow
            x_position = col * (img_width + 2) + img_width // 2
        
            axes[0].text(
                x_position, -20, f"Value: {value:.2f}",
                fontsize=10, color='white', ha='center', va='top',
                transform=axes[0].transData
            )
            axes[1].text(
                x_position, -20, f"Value: {value:.2f}",
                fontsize=10, color='white', ha='center', va='top',
                transform=axes[1].transData
            )
        
        plt.tight_layout(pad=3.0) # Add more padding between subplots
        plt.show()
        
        if log_dir:
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved tweaked latents to {file_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error tweaking latent dimension {latent_dim}: {e}")



def plot_reconstructions(
        beta_vae_model, data_loader, timestep=0.0, class1=0, 
        num_samples=8, nrow=4, use_latents=True, filename='reconstructed_samples.png',
        log_dir=None, device='cuda'
    ) -> None:
    """Plot the original and reconstructed images"""

    beta_vae_model.eval()
    testloader = data_loader.val_dataloader(timestep=timestep)
    samples_dict = collect_samples(
        dataloader=testloader, target_classes=[class1], 
        num_samples=num_samples, use_latents=use_latents, max_batches=100
    )
    
    base_imgs1, class_0_indices = samples_dict[class1]

    try:
        # Reconstruct the base images
        with torch.no_grad():
            loss_dict, recon_batch, z, mu, logvar = beta_vae_model(base_imgs1.to(device))[0]
            imgs = torch.cat([base_imgs1, recon_batch], dim=0)

            # Denormalize and clamp to [0, 255]
            imgs = imgs.detach().cpu()
            imgs = denorm_tensor(imgs)
            grid_img = torchvision.utils.make_grid(imgs, nrow=nrow, padding=0)
        
        plt.figure(figsize=(15, 15))
        plt.imshow(grid_img.permute(1, 2, 0).numpy())
        plt.title(f'Original and Reconstructed Images of Class {class1}', fontsize=12)

        plt.axis('off')
        plt.show()

        if log_dir:
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved interpolations to {file_path}")
        plt.close()
    
    except Exception as e:
        logging.error(f"Error plotting reconstructions: {e}")



def plot_reconstruction_error(
    timesteps: list[float],
    mse_values: list[float],
    psnr_scores: list[float],
    filename: str = 'reconstruction_error.png',
    mse_title: str = 'Mean Squared Error vs. Timestep',
    psnr_title: str = 'Peak Signal-to-Noise Ratio vs. Timestep',
    xlabel: str = 'Embedded Timestep (t)',
    log_dir: Optional[str] = None,
    window_size: int = 5,
    invert_xaxis: bool = True,
    use_wandb: bool = False
):
    """Plot the MSE and PSNR values per timestep with inverted x-axis and moving average background."""
    try:
        # Moving average of MSE and PSNR
        mse_moving_avg = np.convolve(mse_values, np.ones(window_size) / window_size, mode='same')
        psnr_moving_avg = np.convolve(psnr_scores, np.ones(window_size) / window_size, mode='same')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # MSE Plot
        sns.lineplot(x=timesteps, y=mse_values, marker='o', ax=axes[0], color='blue', label="MSE")
        axes[0].fill_between(timesteps, mse_moving_avg, color='lightgrey', alpha=0.3, label="MSE Moving Avg")
        axes[0].set_title(mse_title)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('MSE Value')
        if invert_xaxis:  # Check if x-axis inversion is needed
            axes[0].invert_xaxis()  # Start from 1.0 to 0.0
        axes[0].legend()

        # PSNR Plot
        sns.lineplot(x=timesteps, y=psnr_scores, marker='o', ax=axes[1], color='orange', label="PSNR")
        axes[1].fill_between(timesteps, psnr_moving_avg, color='lightblue', alpha=0.3, label="PSNR Moving Avg")
        axes[1].set_title(psnr_title)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('PSNR Score')
        if invert_xaxis:  # Check if x-axis inversion is needed
            axes[1].invert_xaxis()  # Start from 1.0 to 0.0
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        if log_dir:
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved MSE and PSNR plot to {file_path}")
        
        if use_wandb:
            wandb.log({'MSE vs. PSNR': [wandb.Image(plt)]})

        plt.close(fig)

    except Exception as e:
        logging.error(f"Error plotting reconstruction error: {e}")



def plot_images_grid(real_imgs, fake_imgs, nrow=None, title='Comparison of Original and Reconstructed Images', file_name='recon_samples.png', log_dir=None, use_wandb=False):
    """Plot a grid of two types of images (e.g. real vs. reconstructed)"""
    try: 
        # Denormalize images to [0, 255]
        batch_size = real_imgs.size(0)
        real_imgs = denorm_tensor(real_imgs.detach().cpu())
        fake_imgs = denorm_tensor(fake_imgs.detach().cpu())

        min_batch_size = min(real_imgs.size(0), fake_imgs.size(0))
        real_imgs, fake_imgs = real_imgs[:min_batch_size], fake_imgs[:min_batch_size]

        # Concat along batch dim.
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)                     # (2B, C, H, W)
        
        if nrow is None:
            nrow = min(batch_size, len(all_imgs) // 2)                                  # Default to half the batch size

        grid_img = torchvision.utils.make_grid(all_imgs, nrow=nrow, padding=0)
        grid_img_np = grid_img.permute(1, 2, 0).numpy()                         # (C, H, W) -> (H, W, C) for display

        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img_np)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.show()
        
        if log_dir:
            file_path = os.path.join(log_dir, file_name)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved image grid to {file_path}")
        
        if use_wandb:
            wandb.log({title: [wandb.Image(grid_img_np, caption=title)]})
            
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting images grid: {e}")





def plot_some_images(images, n_samples=16, nrow=None, title='Sample Images', filename='sample_images.png', log_dir=None, use_wandb=False):
    """Plot a grid of sample images"""
    try: 

        if isinstance(images, list):
            images = torch.cat(images, dim=0)       # concatenate along batch dimension
        
        if nrow is None:
            nrow = len(images) // 2

        # Denormalize and clamp to [0, 255]
        images = images.detach().cpu()
        images = denorm_tensor(images)

        # Make grid of images
        grid_img = torchvision.utils.make_grid(images[:n_samples], nrow=nrow, padding=0)
        grid_img_np = grid_img.permute(1, 2, 0).numpy()

        plt.figure(figsize=(15, 15))
        plt.imshow(grid_img_np)
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.show()

        if log_dir:
            file_path = os.path.join(log_dir, filename)
            plt.savefig(file_path, dpi=300)
            logging.info(f"Saved image to {file_path}")
        
        if use_wandb:
            wandb.log({title: [wandb.Image(plt)]})

        plt.close()
    except Exception as e:
        logging.error(f"Error plotting some images: {e}")




def plot_reconstructions(beta_vae_model, data_loader, device='cuda', use_wandb=False):
    """Plot the original and reconstructed images"""
    beta_vae_model.eval()

    testloader = data_loader['train']
    dataiter = iter(testloader)
    image, labels, latent = next(dataiter)

    # Reconstruct the images
    with torch.no_grad():
        loss_dict, recon_batch, z, mu, logvar = beta_vae_model(latent.to(device))

    original_grid = torchvision.utils.make_grid(latent.cpu(), nrow=8, normalize=True, scale_each=True, padding=0)
    recon_grid = torchvision.utils.make_grid(recon_batch.cpu(), nrow=8, normalize=True, scale_each=True, padding=0)

    original_grid_np = original_grid.permute(1, 2, 0).numpy()
    recon_grid_np = recon_grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid_np)
    plt.title('Original Images', fontsize=10)
    plt.axis('off')

    # Plot the reconstructed images
    plt.subplot(1, 2, 2)
    plt.imshow(recon_grid_np)
    plt.title('Reconstructed Images', fontsize=10)
    plt.axis('off')

    plt.show()


def plot_inverse_diffusion(
    all_latents: List[torch.Tensor],
    timesteps: List[float],
    nrow: int = None,
    title: str = 'Interpolated Latent Codes',
    filename: str = 'latent_codes.png',
    log_dir: str = None,
    use_wandb=False
):
    """
    Plot the inverse diffusion process from timestep=1.0 (data) to timestep=0.0 (fully noised data).
    Each column represents a timestep, each row is one image from the batch.
    """
    print(f"Selected timesteps for plotting: {timesteps}")
    
    all_latents = [denorm_tensor(latent) for latent in all_latents]
    batch_size = all_latents[0].shape[0]

    if nrow is None:
        nrow = min(batch_size, len(timesteps))

    images_to_grid = []
    for i in range(min(nrow, batch_size)):
        time_sequence = [latents[i:i+1] for latents in all_latents]  # Take single image from each timestep
        row_images = torch.cat(time_sequence, dim=0)                 # Concatenate along batch dimension
        images_to_grid.append(row_images)

    all_latents = torch.cat(images_to_grid, dim=0)
    grid_img = torchvision.utils.make_grid(all_latents, nrow=nrow, padding=0)
    grid_img_np = grid_img.cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 15))
    plt.imshow(grid_img_np)
    plt.title(title, fontsize=16)

    ax = plt.gca()
    column_width = grid_img_np.shape[1] / nrow
    x_positions = [column_width * (i + 0.5) for i in range(nrow)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"t={t:.2f}" for t in timesteps], rotation=0, fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, filename)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        logging.info(f"Saved inverse diffusion process to {file_path}")
    
    if use_wandb:
        wandb.log({title: [wandb.Image(plt)]})

    plt.close()


# ----------------------------------------------
# Plotting functions for Interpolation Analysis
# ----------------------------------------------
def plot_metrics(data: Any, n_batches: int = 10, log_dir: str = None):
    """
    Plot the metrics (MSE and PSNR) for the interpolation analysis using Seaborn.
    
    Parameters:
    - data: DataFrame containing 'Metric', 'DDIM Steps', 'Timestep', 'Value'.
    """
    df = pd.DataFrame(data)
    mse_data = df[df['Metric'] == 'MSE']
    psnr_data = df[df['Metric'] == 'PSNR']

    # Define a consistent palette for the DDIM Steps
    unique_steps = sorted(df['DDIM Steps'].unique())
    palette = sns.color_palette("viridis", n_colors=len(unique_steps))
    step_palette = dict(zip(unique_steps, palette))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True)

    # Plot MSE
    sns.lineplot(
        ax=axes[0],
        data=mse_data,
        x='Timestep',
        y='Value',
        hue='DDIM Steps',
        marker='o',
        palette=step_palette  # Use the consistent palette
    )
    # Add mean line for MSE
    mse_mean = mse_data.groupby('Timestep')['Value'].mean().reset_index()
    axes[0].plot(
        mse_mean['Timestep'], 
        mse_mean['Value'], 
        linestyle='--', 
        color='orange', 
        label='Mean MSE'
    )
    axes[0].set_title(f'Reconstruction Error (MSE) over {n_batches} Batches', fontsize=14)
    axes[0].set_xlabel('Timesteps (1.0 = Original, 0.0 = Full Noise)', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_xlim(0.8, 0.0)
    axes[0].legend(title='DDIM Steps and Mean')
    axes[0].grid(alpha=0.3)

    # Plot PSNR
    sns.lineplot(
        ax=axes[1],
        data=psnr_data,
        x='Timestep',
        y='Value',
        hue='DDIM Steps',
        marker='o',
        palette=step_palette  # Use the same consistent palette
    )
    # Add mean line for PSNR
    psnr_mean = psnr_data.groupby('Timestep')['Value'].mean().reset_index()
    axes[1].plot(
        psnr_mean['Timestep'], 
        psnr_mean['Value'], 
        linestyle='--', 
        color='red', 
        label='Mean PSNR'
    )
    axes[1].set_title(f'Reconstruction Quality (PSNR) over {n_batches} Batches', fontsize=14)
    axes[1].set_xlabel('Timesteps (1.0 = Original, 0.0 = Full Noise)', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_xlim(0.8, 0.0)
    axes[1].legend(title='DDIM Steps and Mean')
    axes[1].grid(alpha=0.3)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.show()
    
    if log_dir:        
        # Save plot to file
        json_path = os.path.join(log_dir, 'reconstruction-metrics.json')
        df.to_json(json_path, orient='records')
        logging.info(f"Saved reconstruction metrics to {json_path}")

        # Save plot to file
        file_path = os.path.join(log_dir, 'reconstruction-metrics.png')
        plt.savefig(file_path, dpi=300)
        logging.info(f"Saved reconstruction metrics to {file_path}")
    
    plt.close(fig)




if __name__ == '__main__':
    pass