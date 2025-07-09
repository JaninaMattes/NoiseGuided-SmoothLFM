import os, sys
import random
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt, rcParams


import gc
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import torch.optim as optim


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm

import os


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# helper Jutils imports
from jutils import denorm
from jutils import ims_to_grid
from jutils.vision import tensor2im
from jutils import exists, freeze, default
from jutils import tensor2im, ims_to_grid



# Setup project root for import resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)

from ldm.helpers import un_normalize_ims # Convert from [-1, 1] to [0, 255]
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor
from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule


torch.set_float32_matmul_precision('high')







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




############################################################################
#                   Custom Dataset for PCA samples                        #
############################################################################
class PCADataset(Dataset):
    def __init__(self, pca_latents, labels=None):
        self.pca_latents = pca_latents
        self.labels = labels

    def __len__(self):
        return len(self.pca_latents)

    def __getitem__(self, idx):
        item = {'pca': torch.tensor(self.pca_latents[idx], dtype=torch.float32)}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    
    
    
def create_pca_dataloader(pca_latents, labels=None, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for PCA latents.
    
    Args:
        pca_latents (np.ndarray): PCA latent vectors.
        labels (np.ndarray, optional): Corresponding labels for the latents.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        DataLoader: A DataLoader instance for the PCA dataset.
    """
    pca_dataset = PCADataset(pca_latents, labels)
    return DataLoader(pca_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    

############################################################################
#                   Linear Probe for Classifier Accuracy                    #
############################################################################
""" Linear Probe for β-VAE PCA Features"""
class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)





def train_linear_probe(
    linear_probe,
    train_loader,
    val_loader,
    source_timestep,
    target_timestep,
    label_key='label',
    latent_key='pca',
    device='cuda',
    epochs=50,
    patience=7,
    lr=1e-3,
    output_csv='linear_probe_metrics.csv',
    beta_value=1e-4,
    model_name='',
):
    linear_probe = linear_probe.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.parameters(), lr=lr)

    history = []
    best_val_acc = -float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        linear_probe.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", leave=False):
            pca_vectors = batch[latent_key].to(device)
            labels = batch[label_key].to(device).view(-1)

            # Check if labels are within valid range
            if (labels < 0).any() or (labels >= linear_probe.linear.out_features).any():
                print(f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}")
                continue

            # Extra safeguard to ensure 1D shape
            if labels.ndim > 1:
                print(f"[DEBUG] Reshaping labels from {labels.shape} to 1D")
                labels = labels.squeeze()
                             
            logits = linear_probe(pca_vectors)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total

        # Validation loop
        linear_probe.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}", leave=False):
                pca_vectors = batch[latent_key].to(device)
                labels = batch[label_key].to(device)

                # Check if labels are within valid range
                if (labels < 0).any() or (labels >= linear_probe.linear.out_features).any():
                    print(f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}")
                    continue

                # Extra safeguard to ensure 1D shape
                if labels.ndim > 1:
                    print(f"[DEBUG] Reshaping labels from {labels.shape} to 1D")
                    labels = labels.squeeze()
                
                logits = linear_probe(pca_vectors)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss /= val_total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        history.append({
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Train_Accuracy': train_acc,
            'Val_Loss': val_loss,
            'Val_Accuracy': val_acc,
            'Precision': precision,
            'Recall': recall,
            'Beta': beta_value,
            'Model': model_name,
            'Source_Timestep': source_timestep,
            'Target_Timestep': target_timestep,
        })

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] Stopped at epoch {epoch+1}")
                break

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    df = pd.DataFrame(history)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved: {output_csv}")

    return df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

############################################################################
#                 Plot Validation Accuracy Curve (Nicer Style)             #
############################################################################
def plot_validation_curve(df_metrics, source_timestep, target_timestep, beta, save_path=None):
    """
    Plot a polished validation accuracy curve using seaborn and custom style.
    """
    # Set global style settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Cambria", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.facecolor"] = "#f5f5f5"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.color"] = "grey"
    plt.rcParams["axes.grid"] = True

    epochs = df_metrics['Epoch']
    val_acc = df_metrics['Val_Accuracy']

    fig, ax = plt.subplots(figsize=(10, 6))
    color = sns.color_palette("crest", n_colors=1)[0]

    ax.plot(epochs, val_acc, marker='o', color=color, linewidth=2.5, markersize=7, label="Validation Accuracy")

    # Annotate final accuracy
    ax.text(
        epochs.values[-1], val_acc.values[-1] + 0.01,
        f"{val_acc.values[-1]*100:.1f}%", ha='center', fontsize=11, color=color
    )

    ax.set_title(
        rf"Validation Accuracy - β-VAE   (source={source_timestep:.2f} → target={target_timestep:.2f},  β={beta})",
        fontsize=16, pad=20
    )
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()

    if save_path is None:
        save_path = f"validation_curve_{source_timestep:.2f}_{target_timestep:.2f}_beta{beta}.png"
        print(f"[INFO] No save path provided, using default: {save_path}")

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Validation curve saved to: {save_path}")

    plt.show()
    plt.close()

    
    

############################################################################
#                       Plot UMAP 2D Cluster Plot                          #
############################################################################
def plot_umap_pca(
    pca_latents,
    labels=None,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    max_samples=50000,
    save_to_path=None,
    title="UMAP projection of PCA latents",
    figsize=(10, 8)
):
    """
    Plot a UMAP projection of PCA-projected latents, optionally colored by labels.

    Args:
        pca_latents (np.ndarray): PCA-projected latents, shape (N, D)
        labels (np.ndarray, optional): Labels for coloring
        n_neighbors (int): UMAP parameter for local neighborhood
        min_dist (float): UMAP parameter for minimum distance
        n_components (int): Number of UMAP components (usually 2)
        max_samples (int): Max number of samples to plot
        save_to_path (str, optional): If provided, save plot to this path
        title (str): Plot title
        figsize (tuple): Figure size
    """
    if pca_latents.shape[0] > max_samples:
        pca_latents = pca_latents[:max_samples]
        labels = labels[:max_samples] if labels is not None else None

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    embedding = reducer.fit_transform(pca_latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab20', s=5, alpha=0.7
        )
        plt.colorbar(scatter, label='Class label')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] UMAP plot saved to: {save_to_path}")
    plt.show()



def run_pca_eval(
    source_timestep=0.20, 
    target_timestep=1.00, 
    beta=1.0, 
    dataset_name='imagenet256-dataset', 
    group="validation", 
    checkpoint=None, 
    data_path=None, 
    project_name=None, 
    model_name=None, 
    num_components=5, 
    max_samples=10000, 
    batch_size=32, 
    device=None, 
    results_root="results"
):
    """
    Full pipeline to run PCA on bottleneck latents, remap labels to contiguous range,
    train a linear probe, save metrics, components, and plots.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load VAE model
    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(checkpoint, map_location='cpu')
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

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
        group_name=group,
    )
    data.setup(stage="fit" if group == "validation" else "test")
    dataloader = data.val_dataloader() if group == "validation" else data.test_dataloader()

    # Results directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_results_dir = Path(results_root) / project_name / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results saved to: {base_results_dir}")

    ########################################################################
    #                   Collect bottleneck latents                        #
    ########################################################################
    with torch.no_grad():
        all_latents, all_labels, curr_samples = [], [], 0
        print("\n--- Collecting bottleneck latents ---")
        for batch in tqdm(dataloader, desc="Collecting latents"):
            if curr_samples >= max_samples:
                break
            latents = batch[f'latents_{source_timestep:.2f}'].to(device).cpu()
            all_latents.append(latents)
            if 'label' in batch:
                all_labels.append(batch['label'].cpu())
            curr_samples += latents.shape[0]

    all_latents = torch.cat(all_latents, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None
    print(f"[INFO] Collected latents shape: {all_latents.shape}")

    # Flatten
    all_latents = all_latents.reshape(all_latents.shape[0], -1)
    print(f"[INFO] Latents reshaped to: {all_latents.shape}")

    ########################################################################
    #                       Fit PCA and project                           #
    ########################################################################
    print(f"[INFO] Fitting PCA with {num_components} components ...")
    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)
    print(f"[INFO] Explained variance: {pca.explained_variance_ratio_}")

    ########################################################################
    #                  Remap labels to contiguous range                   #
    ########################################################################
    unique_labels = np.unique(all_labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    all_labels_mapped = np.vectorize(label_map.get)(all_labels)
    num_classes = len(unique_labels)
    print(f"[INFO] Found {num_classes} unique classes. Labels remapped to 0–{num_classes - 1}.")

    ########################################################################
    #                  Split into train and val sets                      #
    ########################################################################
    X_train, X_val, y_train, y_val = train_test_split(
        pca_latents, all_labels_mapped, test_size=0.2, random_state=42, stratify=all_labels_mapped
    )
    train_loader = DataLoader(PCADataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(PCADataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4)

    for loader in [train_loader, val_loader]:
        for batch in loader:
            assert 'pca' in batch, "Expected 'pca' key in batch for LinearProbe"

    print(f"[INFO] Train set size: {len(train_loader.dataset)}, Val set size: {len(val_loader.dataset)}")
    
    ########################################################################
    #                        Train linear probe                           #
    ########################################################################
    print("\n--- Training Linear Probe ---")
    linear_probe = LinearProbe(hidden_size=num_components, num_classes=num_classes)
    output_csv = base_results_dir / "linear_probe_metrics.csv"

    df_metrics = train_linear_probe(
        linear_probe=linear_probe,
        train_loader=train_loader,
        val_loader=val_loader,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        device=device,
        epochs=50,
        patience=10,
        lr=1e-4,
        output_csv=str(output_csv),
        beta_value=beta,
        model_name=model_name,
    )

    ########################################################################
    #                  Plot and save validation curve                     #
    ########################################################################
    plot_path = base_results_dir / "val_accuracy_curve.png"
    plot_validation_curve(df_metrics, save_path=plot_path, source_timestep=source_timestep, target_timestep=target_timestep, beta=beta)

    ########################################################################
    #                Save PCA projections and components                 #
    ########################################################################
    np.save(base_results_dir / "pca_latents.npy", pca_latents)
    np.save(base_results_dir / "pca_labels.npy", all_labels_mapped)
    np.save(base_results_dir / "pca_components.npy", pca.components_)
    np.save(base_results_dir / "explained_variance.npy", pca.explained_variance_ratio_)

    print("\n[INFO] PCA evaluation completed.")
    print(f"[INFO] Results directory: {base_results_dir}")
    print(f"[INFO] Metrics CSV: {output_csv}")
    print(f"[INFO] Validation curve plot: {plot_path}")

    ########################################################################
    #                  Plot UMAP projection                               #
    ########################################################################
    print("\n--- Plotting UMAP Projection ---")
    plot_umap_pca(
        pca_latents=pca_latents, 
        labels=all_labels_mapped, 
        n_components=2, 
        max_samples=50000, 
        save_to_path=base_results_dir / "umap_plot.png", 
        title="UMAP of PCA-projected Latents"
    )

    print("\n[INFO] PCA evaluation completed.")
    print(f"[INFO] Results directory: {base_results_dir}")
    print(f"[INFO] Metrics CSV: {output_csv}")
    print(f"[INFO] Validation curve plot: {plot_path}")

    return df_metrics, pca_latents, all_labels_mapped, pca.components_, pca.explained_variance_ratio_





# if __name__ == "__main__":

#     #####################################
#     # Evaluation Setup
#     #####################################
#     source_timestep = 0.20
#     target_timestep = 1.00
#     beta            = 1e-4  # Beta value for the VAE
#     dataset_name    = 'imagenet256-dataset-T000006'
#     group           = "validation"  # or "test"
#     baseline        = (source_timestep == 0.50 and target_timestep == 0.50)

#     num_components   = 5
#     max_samples      = 80000
#     batch_size       = 64



#     #####################################
#     # Model Setup
#     #####################################
    
#     # beta: 1e-4 
#     Beta02x10x_1e4b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'
    
#     # beta: 0.1
#     Beta00x00x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt'
#     Beta02x02x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt'                     # Open 
#     Beta05x05x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt'                     # Open (Baseline)
#     Beta05x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-02/101646/checkpoints/last.ckpt'                                       # Open
#     Beta04x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.40x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'  
#     Beta03x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.30x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'  
#     Beta02x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt'                    ####### DONE
#     Beta00x10x_01b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-1.00x-0.1b/2025-06-18/29852/V0-eV2/2025-06-24/29852/checkpoints/last.ckpt'                 # Open

#     # beta: 0.5
#     Beta02x10x_05b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt'

#     # beta: 1.0
#     Beta05x05x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
#     Beta05x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'                                                                                                                                   # Open
#     Beta02x10x_1b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'                                          # Open

#     # beta: 2.0
#     Beta02x10x_2b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-2.0b/V0/2025-07-02/101646/checkpoints/last.ckpt'                     # Open
    
    
#     # beta: 3.0
#     Beta02x10x_3b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt'                     # Open

#     # beta: 5.0
#     Beta05x05x_5b ='./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt'                                                                                                                                   # Open (Baseline)
#     Beta05x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'                                                                                                                                  # Open
#     Beta02x10x_5b = './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'                   # Open



if __name__ == "__main__":
    #####################################
    # Shared Parameters
    #####################################
    dataset_name    = 'imagenet256-dataset-T000006'
    group           = "validation"
    num_components  = 5
    max_samples     = 80000
    batch_size      = 64
    data_path       = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5'

    #####################################
    # Define ALL models with target = 1.0
    #####################################
    model_configs = [
        # beta: 1e-4
        {"name": "Beta02x10x_1e4b", "beta": 1e-4, "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},

        # beta: 0.1
        {"name": "Beta05x10x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-02/101646/checkpoints/last.ckpt'},
        {"name": "Beta04x10x_01b",  "beta": 0.1,  "source_ts": 0.40, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.40x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},
        {"name": "Beta03x10x_01b",  "beta": 0.1,  "source_ts": 0.30, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.30x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_01b",  "beta": 0.1,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt'},
        {"name": "Beta00x10x_01b",  "beta": 0.1,  "source_ts": 0.00, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-1.00x-0.1b/2025-06-18/29852/V0-eV2/2025-06-24/29852/checkpoints/last.ckpt'},

        # beta: 0.5
        {"name": "Beta02x10x_05b",  "beta": 0.5,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt'},

        # beta: 1.0
        {"name": "Beta05x10x_1b",   "beta": 1.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_1b",   "beta": 1.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'},

        # beta: 2.0
        {"name": "Beta02x10x_2b",   "beta": 2.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-2.0b/V0/2025-07-02/101646/checkpoints/last.ckpt'},

        # beta: 3.0
        {"name": "Beta02x10x_3b",   "beta": 3.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt'},

        # beta: 5.0
        {"name": "Beta05x10x_5b",   "beta": 5.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_5b",   "beta": 5.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'},
    ]

    #####################################
    # Run each model
    #####################################
    all_metrics = []

    for config in model_configs:
        beta        = config["beta"]
        checkpoint  = config["ckpt"]
        source_ts   = config["source_ts"]
        target_ts   = config["target_ts"]
        model_tag   = config["name"]
        project_name = "Test_BetaVAE_PCA_Quantitative_Eval"
        model_name  = f"{model_tag}_{dataset_name}"

        print(f"\n🔧 [INFO] Running model: {model_tag} (β={beta}, source={source_ts:.2f}, target={target_ts:.2f})")

        df_metrics, *_ = run_pca_eval(
            source_timestep=source_ts,
            target_timestep=target_ts,
            beta=beta,
            dataset_name=dataset_name,
            group=group,
            checkpoint=checkpoint,
            data_path=data_path,
            project_name=project_name,
            model_name=model_name,
            num_components=num_components,
            max_samples=max_samples,
            batch_size=batch_size,
            results_root="results"
        )

        df_metrics["Model"]         = model_tag
        df_metrics["Beta"]          = beta
        df_metrics["Source_TS"]     = source_ts
        df_metrics["Target_TS"]     = target_ts
        all_metrics.append(df_metrics)


    #####################################
    # Combined plot
    #####################################
    import seaborn as sns
    import matplotlib.pyplot as plt

    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)

        plt.figure(figsize=(12, 7))
        sns.set(style="whitegrid", font="serif")
        palette = sns.color_palette("crest", n_colors=len(combined_df["Model"].unique()))

        for model_tag, color in zip(combined_df["Model"].unique(), palette):
            model_df = combined_df[combined_df["Model"] == model_tag]
            plt.plot(model_df["Epoch"], model_df["Val_Accuracy"], marker='o', linewidth=2, label=f"{model_tag}", color=color)

        plt.title(f"Validation Accuracy Comparison (target = 1.00)", fontsize=16, pad=20)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Validation Accuracy", fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(title="Model", fontsize=9, loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)

        combined_plot_path = f"combined_validation_curve_target1.00.png"
        plt.savefig(combined_plot_path, bbox_inches='tight', dpi=300)
        print(f"\n[INFO] Combined validation plot saved to: {combined_plot_path}")
        plt.show()
