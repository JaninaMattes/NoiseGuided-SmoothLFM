import os
import sys
import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from lightning import seed_everything

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    precision_score,
    recall_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

import json
import random
from pathlib import Path


import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

import umap

# Helper utilities
from jutils import denorm, ims_to_grid, exists, freeze, default
from jutils.vision import tensor2im

# Project root path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
sys.path.append(project_root)

# Project-specific modules
from ldm.helpers import un_normalize_ims
from data_processing.tools.norm import denorm_metrics_tensor, denorm_tensor
from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

# Torch precision
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
        encoded = module.model.encode(source_latents)
        latents = encoded['latent_dist'].mode()
        all_latents.append(latents.detach().cpu().numpy())

    combined_latents = np.vstack(all_latents)
    print(f"[INFO] Collected {combined_latents.shape[0]} latent vectors of dim {combined_latents.shape[1]}.")

    # Sorted by vairance (highest --> lowest)
    pca = PCA(n_components=num_components)
    pca.fit(combined_latents)

    print(f"[INFO] PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"[INFO] Total explained: {np.sum(pca.explained_variance_ratio_):.2f}")

    return pca.components_, pca.explained_variance_ratio_




############################################################################
#                   Visualise most Important PCA Vectors                   #
############################################################################

def plot_pca_2d_projection(pca_latents, labels=None, save_path=None, title="PCA 2D Projection"):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(
            pca_latents[:, 0], pca_latents[:, 1], # first and second PCA components
            c=labels, cmap='tab20', s=5, alpha=0.7
        )
        plt.colorbar(scatter, label='Class label')
    else:
        plt.scatter(pca_latents[:, 0], pca_latents[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] PCA 2D projection saved to: {save_path}")

    plt.show()





def plot_latent_histograms(pca_latents, num_bins=50, save_dir=None):
    num_components = pca_latents.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(num_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in range(num_components):
        ax = axes[i]
        ax.hist(pca_latents[:, i], bins=num_bins, color='cornflowerblue', alpha=0.7)
        ax.set_title(f"PCA Component {i+1}")
        ax.set_yticks([])
        ax.set_xticks([])

    # Turn off unused axes
    for ax in axes[num_components:]:
        ax.axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "pca_latent_histograms.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] Histograms saved to: {save_path}")

    plt.show()





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
        self.output_dim = num_classes

    def forward(self, x):
        return self.linear(x)

    def get_output_dim(self):
        return self.output_dim



class TwoLayerProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.output_dim = num_classes

    def forward(self, x):
        return self.linear(x)


    def get_output_dim(self):
        return self.output_dim



#############################################################################
#                       Custom Probe Trainer Module                       #
##############################################################################

def train_linear_probe(
    linear_probe,
    train_loader,
    val_loader,
    source_timestep,
    target_timestep,
    label_key='label',
    latent_key='pca',
    device=None,
    epochs=500,
    patience=10,
    lr=1e-4,
    output_csv='linear_probe_metrics.csv',
    beta_value=1e-4,    # default low beta value for β-VAE
    model_name='',
    output_dim=90
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear_probe = linear_probe.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.parameters(), lr=lr) # use AdamW for better generalization

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
            if (labels < 0).any() or (labels >= linear_probe.get_output_dim()).any():
                print(f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}")
                continue

            # Extra safeguard to ensure 1D shape
            if labels.ndim > 1:
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
                if (labels < 0).any() or (labels >= linear_probe.output_dim).any():
                    print(f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}")
                    continue

                # Extra safeguard to ensure 1D shape
                if labels.ndim > 1:
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
            'Beta'


            : beta_value,
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




############################################################################
#                 Plot Validation Accuracy Curve (Nicer Style)             #
############################################################################
def set_plot_style():
    """
    Sets consistent global style for plots (fonts, grid, colors).
    """
    # Set global style to match example
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,  # Set to True if you want full LaTeX rendering
        "axes.facecolor": "#e8ecf0",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.2,
    })
    sns.set_palette("Set2")  # Ensures consistent coloring for all plots


def plot_validation_curve(df_metrics, source_timestep, target_timestep, beta, save_path=None):
    """
    Plot a polished validation accuracy curve using seaborn and consistent style.
    """
    set_plot_style()

    epochs = df_metrics['Epoch']
    val_acc = df_metrics['Val_Accuracy']

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Use the first color from the Set2 palette
    color = sns.color_palette("Set2", n_colors=1)[0]

    ax.plot(epochs, val_acc, marker='o', color=color, linewidth=2.5, markersize=7, label="Validation Accuracy")

    ax.text(
        epochs.values[-1], val_acc.values[-1] + 0.01,
        f"{val_acc.values[-1]*100:.1f}%", ha='center', fontsize=11, color=color
    )

    ax.set_title(
        rf"Validation Accuracy - β-VAE   (source={source_timestep:.2f} → target={target_timestep:.2f},  β={beta})",
        fontsize=16, pad=20
    )

    ymax = math.ceil((val_acc.max() + 0.05) * 20) / 20
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="upper right", title="Model", title_fontsize=13, frameon=False)

    plt.tight_layout()

    if save_path is None:
        save_path = f"validation_curve_{source_timestep:.2f}_{target_timestep:.2f}_beta{beta}.png"
        print(f"[INFO] No save path provided, using default: {save_path}")

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Validation curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_combined_validation_curve(df_combined, save_path, source_timestep, target_timestep, beta):
    """
    Plot a combined validation accuracy curve for multiple probes.
    """
    set_plot_style()

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=df_combined['Probe'].nunique())

    ax = sns.lineplot(
        data=df_combined, x='Epoch', y='Val_Accuracy',
        hue='Probe', marker='o', palette=palette
    )

    for probe, sub_df in df_combined.groupby('Probe'):
        last_row = sub_df.iloc[-1]
        ax.text(
            last_row['Epoch'] + 0.2,
            last_row['Val_Accuracy'],
            f"{last_row['Val_Accuracy']*100:.2f}%",
            fontsize=11,
            color=ax.get_lines()[list(df_combined['Probe'].unique()).index(probe)].get_color(),
            weight='bold'
        )

    ymax = math.ceil((df_combined['Val_Accuracy'].max() + 0.05) * 20) / 20
    plt.ylim(0, ymax)

    plt.title(
        f"Validation Accuracy Comparison (β={beta}, source={source_timestep:.2f} → target={target_timestep:.2f})",
        fontsize=16
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=14)
    plt.legend(title="Probe Type", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Combined validation curve saved to: {save_path}")
    plt.show()
    plt.close()




############################################################################
#                       Plot UMAP 2D Cluster Plot                          #
############################################################################
def plot_umap_pca(
    pca_latents, # PCA-projected latents, shape (N, D)
    labels=None,
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    max_data_samples=100000,
    save_to_path=None,
    title="UMAP projection of PCA latents",
    figsize=(10, 8),
    random_state=42 # state of life s
):
    """
    Plot a UMAP projection of PCA-projected latents, optionally colored by labels.

    Args:
        pca_latents (np.ndarray): PCA-projected latents, shape (N, D)
        labels (np.ndarray, optional): Labels for coloring
        n_neighbors (int): UMAP parameter for local neighborhood
        min_dist (float): UMAP parameter for minimum distance
        n_components (int): Number of UMAP components (usually 2)
        max_data_samples (int): Max number of samples to plot
        save_to_path (str, optional): If provided, save plot to this path
        title (str): Plot title
        figsize (tuple): Figure size
    """
    set_plot_style()

    if pca_latents.shape[0] > max_data_samples:
        pca_latents = pca_latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None
        print(f"[INFO] PCA shape reduced to {pca_latents.shape[0]} samples for plotting.")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
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
    plt.close()



def plot_pca_2d(latents, labels=None, max_data_samples=100000, save_to_path=None, title="2D PCA Projection", figsize=(10, 8)):
    """ Plot a 2D PCA projection of latents, optionally colored by labels.
    """

    set_plot_style()

    if latents.shape[0] > max_data_samples:
        latents = latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=5, alpha=0.7)
        plt.colorbar(scatter, label='Class label')
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] PCA plot saved to: {save_to_path}")
    plt.show()
    plt.close()



def plot_tsne_2d(latents, labels=None, max_data_samples=10000, perplexity=30, save_to_path=None, title="t-SNE Projection", figsize=(10, 8)):
    """ Plot a 2D t-SNE projection of latents, optionally colored by labels.
    """

    set_plot_style()


    if latents.shape[0] > max_data_samples:
        latents = latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
    reduced = tsne.fit_transform(latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=5, alpha=0.7)
        plt.colorbar(scatter, label='Class label')
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] t-SNE plot saved to: {save_to_path}")
    plt.show()
    plt.close()




def plot_kmeans_grid(pca_latents, k_values=[2, 3, 5, 10], max_data_samples=100000, figsize_per_plot=(5, 5), save_to_path=None):
    """
    Plot k-Means clustering results over PCA-reduced latents (assumed to be already PCA'd).

    Args:
        pca_latents: Already PCA-reduced latents (e.g., shape [N, D])
        k_values: List of k values to try for k-means
        max_data_samples: Subsample limit for performance
        figsize_per_plot: Size per subplot
        save_to_path: If provided, saves the resulting figure
    """
    set_plot_style()

    if pca_latents.shape[0] > max_data_samples:
        pca_latents = pca_latents[:max_data_samples]

    assert pca_latents.shape[1] >= 2, "Need at least 2 PCA components"

    reduced = pca_latents[:, :2]  # Only use first two PCA components

    n_rows = 2
    n_cols = int(np.ceil(len(k_values) / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0]*n_cols, figsize_per_plot[1]*n_rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42)
        preds = kmeans.fit_predict(pca_latents)  # Cluster in full PCA space

        ax = axes[idx]
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap='tab20', s=5, alpha=0.7)
        ax.set_title(f"k-Means Clustering (k={k})")
        ax.axis("off")

    # Hide any extra unused subplots
    for ax in axes[len(k_values):]:
        ax.axis("off")

    plt.suptitle("k-Means Cluster Projections (PCA-reduced)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_to_path:
        plt.savefig(save_to_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] k-Means grid plot saved to: {save_to_path}")

    plt.show()
    plt.close()




def plot_umap_grid(model_results, group_name, n_neighbors=15, min_dist=0.1, n_components=2, save_path=None, max_data_samples=50000, random_state=None):
    """
    Plot UMAP plots for multiple models in a grid.

    Args:
        model_results (list of dict): Each dict has 'latents', 'labels', 'name'.
        group_name (str): Name of the group (for title).
        save_path (str, optional): If provided, saves the grid plot.
        max_data_samples (int): Maximum samples to plot per model.
    """
    set_plot_style()

    num_models = len(model_results)
    n_cols = 3  # You can adjust
    n_rows = int(np.ceil(num_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for idx, model_info in enumerate(model_results):
        row = idx // n_cols
        col = idx % n_cols

        latents = model_info["latents"]
        labels = model_info["labels"]
        name    = model_info["name"]

        if latents.shape[0] > max_data_samples:
            latents = latents[:max_data_samples]
            labels = labels[:max_data_samples] if labels is not None else None

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(latents)

        ax = axes[row][col]
        if labels is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=3, alpha=0.7)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.7)

        ax.set_title(name, fontsize=12)
        ax.axis("off")

    # Remove unused axes
    for ax in axes.flatten()[num_models:]:
        ax.axis("off")

    plt.suptitle(f"UMAP Comparison — {group_name}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] UMAP grid plot saved to: {save_path}")

    plt.show()
    plt.close()





import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_probe_accuracy_grid(data_df, save_path):
    """
    Plots a grid of boxplots for probe validation accuracy across β-values and PCA settings.

    Parameters:
        data_df (pd.DataFrame): Must contain columns: ['ProbeType', 'PCA', 'Beta', 'ValAccuracies']
        save_path (str or Path): Path to save the figure
    """
    sns.set(style="whitegrid")

    # Convert beta to string for cleaner x-axis
    data_df["BetaStr"] = data_df["Beta"].astype(str)

    # Set up FacetGrid
    g = sns.FacetGrid(
        data_df,
        row="ProbeType",
        col="PCA",
        margin_titles=True,
        sharey=True,
        sharex=False,
        despine=False,
        height=4,
        aspect=1
    )

    def box_with_mean(data, **kwargs):
        # Remove conflicting keys if present
        kwargs.pop("color", None)
        sns.boxplot(
            x="BetaStr",
            y="ValAccuracies",
            data=data,
            color="lightgray",
            fliersize=0,
            **kwargs
        )
        # Overlay red mean line
        means = data.groupby("BetaStr")["ValAccuracies"].mean()
        plt.plot(range(len(means)), means.values, color="red", linewidth=2)

    # Map plotting function
    g.map_dataframe(box_with_mean)

    # Final touches
    g.set_axis_labels("β", "Validation Accuracy")
    g.set_titles(row_template="{row_name}", col_template="PCA = {col_name}")
    g.tight_layout()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Validation Accuracy Distribution by β and PCA Components")
    g.savefig(save_path)
    plt.show()
    plt.close()
    print(f"[INFO] Saved grid plot to: {save_path}")




def plot_probe_accuracy_threepanel(plot_df, pca_num, save_path):
    """
    Creates a 3-panel seaborn grid plot:
    Left: boxplot for Linear Probe
    Middle: boxplot for Two-Layer Probe
    Right: lineplot comparing max accuracy
    """
    sns.set(style="whitegrid")

    # Filter data for specific PCA number
    df = plot_df[plot_df["PCA"] == pca_num].copy()
    df["BetaStr"] = df["Beta"].astype(str)

    # Start figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # --- LEFT: Linear Probe ---
    linear_df = df[df["ProbeType"] == "Linear"]
    sns.boxplot(data=linear_df, x="BetaStr", y="ValAccuracies", ax=axs[0], color="lightgray")
    max_accs = linear_df.groupby("BetaStr")["ValAccuracies"].max()
    axs[0].plot(range(len(max_accs)), max_accs.values, color="red", label="Max Acc", marker="o")
    axs[0].set_title("Linear Probe")
    axs[0].set_xlabel("β")
    axs[0].set_ylabel("Validation Accuracy")
    axs[0].grid(True)

    # --- MIDDLE: Two-Layer Probe ---
    two_df = df[df["ProbeType"] == "Two-Layer"]
    sns.boxplot(data=two_df, x="BetaStr", y="ValAccuracies", ax=axs[1], color="lightgray")
    max_accs_two = two_df.groupby("BetaStr")["ValAccuracies"].max()
    axs[1].plot(range(len(max_accs_two)), max_accs_two.values, color="red", label="Max Acc", marker="o")
    axs[1].set_title("Two-Layer Probe")
    axs[1].set_xlabel("β")
    axs[1].grid(True)

    # --- RIGHT: Combined Line Plot ---
    all_betas = sorted(df["Beta"].unique())
    beta_strs = [Path(b) for b in all_betas]
    max_linear = linear_df.groupby("BetaStr")["ValAccuracies"].max()
    max_two = two_df.groupby("BetaStr")["ValAccuracies"].max()
    axs[2].plot(beta_strs, max_linear[beta_strs], label="Linear Probe", marker="o")
    axs[2].plot(beta_strs, max_two[beta_strs], label="Two-Layer Probe", marker="s")
    axs[2].set_title("Max Accuracy Comparison")
    axs[2].set_xlabel("β")
    axs[2].legend()
    axs[2].grid(True)

    # Final touches
    fig.suptitle(f"Probe Accuracy (PCA={pca_num})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"[INFO] Saved 3-panel plot to {save_path}")
    plt.show()
    plt.close()







######################################################################
#                   Plot Visualizations                              #
######################################################################

def plot_pca_kmeans_scatter_grid(df, project_path="test_outputs"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "axes.facecolor": "#e8ecf0",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.2,
    })

    # Convert types for facetting
    df["PCA"] = df["PCA"].astype(str)
    df["K"] = df["K"].astype(str)

    # Compute global x/y limits with padding
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    # FacetGrid for scatter plots
    g = sns.FacetGrid(df, row="PCA", col="K", margin_titles=True, height=2.8, aspect=1)

    def draw_scatter(data, **kwargs):
        ax = plt.gca()
        ax.set_facecolor("#e8ecf0")
        sns.scatterplot(
            data=data, x="x", y="y", hue="Cluster",
            palette="tab10", s=20, linewidth=0, alpha=0.8, legend=False, ax=ax
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    g.map_dataframe(draw_scatter)

    g.set_axis_labels("PC 1", "PC 2")
    g.set_titles(row_template="PCA = {row_name}", col_template="K = {col_name}")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("K-Means Clustering Scatter Grid across PCA + K", fontsize=14)

    output_path = Path(project_path) / "scatter_grid_pca_kmeans.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved scatter grid to: {output_path}")
    plt.show()
    plt.close()





def plot_probe_val_across_pca(json_path, project_path="test_outputs", probe_filter="Linear"):
    """ Plot validation accuracy across PCA components for a specific probe type.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append({
                    "Model": model["Model"],
                    "Beta": model["Beta"],
                    "SourceTimestep": model["SourceTimestep"],
                    "TargetTimestep": model["TargetTimestep"],
                    "ProbeType": res["ProbeType"],
                    "PCA": res["PCA"],
                    "ValAccuracies": acc
                })

    df = pd.DataFrame(flat_records)
    df = df[df["ProbeType"] == probe_filter]
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)

    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())

    # Global style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "axes.facecolor": "#e8ecf0",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.2,
    })

    # Compute KDE-based global y-limits
    y_vals = df["ValAccuracies"].values
    if len(y_vals) > 1:
        kde = gaussian_kde(y_vals)
        x_range = np.linspace(y_vals.min(), y_vals.max(), 500)
        density = kde(x_range)
        cutoff = x_range[density > max(density) * 0.01]
        y_min = max(0.0, cutoff.min() - 0.1 * (cutoff.max() - cutoff.min()))
        y_max = min(1.2, cutoff.max() + 0.1 * (cutoff.max() - cutoff.min()))
    else:
        y_min, y_max = 0.0, 1.0  # fallback

    # Setup FacetGrid
    g = sns.FacetGrid(df, col="PCA", col_wrap=3, height=2.8, aspect=1.2, sharey=True)

    def draw_minimalist_boxplot(data, **kwargs):
        ax = plt.gca()
        ax.set_facecolor("#e8ecf0")
        data = data.copy()
        data["Beta"] = data["Beta"].astype(float)

        sns.violinplot(
            data=data, x="Beta", y="ValAccuracies",
            ax=ax, order=beta_order,
            inner=None, linewidth=0,
            color="#a0aab8", saturation=0.3
        )

        sns.boxplot(
            data=data, x="Beta", y="ValAccuracies",
            width=0.3, ax=ax, order=beta_order,
            color="white",
            fliersize=1.5, linewidth=0.7,
            boxprops={'facecolor': 'white', 'edgecolor': '#333', 'zorder': 2},
            whiskerprops={'linewidth': 0.7},
            capprops={'linewidth': 0.7},
            medianprops={'color': 'black', 'linewidth': 1}
        )

        # Medians
        medians = data.groupby("Beta", observed=True)["ValAccuracies"].median().reindex(beta_order)
        x_vals = list(range(len(beta_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{b:.4g}" for b in beta_order])
        ax.set_xlabel(r"$\beta$", fontsize=10)
        ax.set_ylabel(r"Validation Accuracy", fontsize=10)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(y_min, y_max)

    g.map_dataframe(draw_minimalist_boxplot)
    g.set_titles(col_template=r"PCA = {col_name}", size=10)

    plt.suptitle(r"Validation Accuracy over Number of PCA Components", fontsize=12, y=1.05)
    plt.tight_layout()

    plot_path = Path(f"{project_path}/pca_beta_boxplot_style_matched.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PCA plot to: {plot_path}")
    plt.show()
    plt.close()





def plot_probe_comparison_grid(json_path, project_path="test_outputs"):
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten data
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append({
                    "Model": model["Model"],
                    "Beta": model["Beta"],
                    "SourceTimestep": model["SourceTimestep"],
                    "TargetTimestep": model["TargetTimestep"],
                    "ProbeType": res["ProbeType"],
                    "PCA": res["PCA"],
                    "ValAccuracies": acc
                })

    df = pd.DataFrame(flat_records)
    df["Beta"] = df["Beta"].astype(float)
    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())
    probe_order = ["Linear", "Two-Layer"]

    # Style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "axes.facecolor": "#e8ecf0",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.2,
    })

    # Create row-wise y-axis limits
    y_limits = df.groupby("ProbeType")["ValAccuracies"].max().to_dict()
    y_limits = {k: min(1.2, v * 1.1) for k, v in y_limits.items()}  # Add margin, clamp at 1.2

    # FacetGrid (don't share y!)
    g = sns.FacetGrid(
        df,
        row="ProbeType", col="PCA",
        height=2.8, aspect=1.2,
        sharey=False,
        row_order=probe_order,
        col_order=pca_order,
        margin_titles=True
    )

    def draw_boxplot_with_medians(data, **kwargs):
        ax = plt.gca()
        probe_type = data["ProbeType"].iloc[0]
        ax.set_facecolor("#e8ecf0")

        sns.violinplot(
            data=data, x="Beta", y="ValAccuracies",
            ax=ax, order=beta_order,
            inner=None, linewidth=0, color="#a0aab8", saturation=0.3
        )

        sns.boxplot(
            data=data, x="Beta", y="ValAccuracies",
            width=0.3, ax=ax, order=beta_order,
            color="white",
            fliersize=1.5, linewidth=0.7,
            boxprops={'facecolor': 'white', 'edgecolor': '#333', 'zorder': 2},
            whiskerprops={'linewidth': 0.7},
            capprops={'linewidth': 0.7},
            medianprops={'color': 'black', 'linewidth': 1}
        )

        medians = data.groupby("Beta", observed=True)["ValAccuracies"].median().reindex(beta_order)
        x_vals = list(range(len(beta_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{b:.4g}" for b in beta_order])
        ax.set_xlabel(r"$\beta$", fontsize=10)
        ax.set_ylabel(r"Validation Accuracy", fontsize=10)
        ax.set_ylim(0, y_limits.get(probe_type, 1.0))
        ax.grid(True, linestyle="--", alpha=0.4)

    g.map_dataframe(draw_boxplot_with_medians)
    g.set_titles(col_template="PCA = {col_name}", row_template="", size=10)

    # Row subtitles
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation"
    }

    g.fig.subplots_adjust(top=0.92, hspace=0.35)
    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = 0.08 if i == len(probe_order) - 1 else 0.035

        g.fig.text(
            center_x, bottom_y - y_offset,
            row_labels[probe],
            fontsize=11, fontweight="bold", ha="center", va="top"
        )

    plt.suptitle(r"Validation Accuracy over Number of PCA Components", fontsize=14, y=1.05)
    plot_path = Path(f"{project_path}/combined_probe_accuracy_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved unified comparison plot to: {plot_path}")
    plt.show()
    plt.close()




def plot_pca_across_betas(json_path, project_path="test_outputs", probe_filter="Linear"):
    import json
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import gaussian_kde

    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten records
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append({
                    "Model": model["Model"],
                    "Beta": model["Beta"],
                    "SourceTimestep": model["SourceTimestep"],
                    "TargetTimestep": model["TargetTimestep"],
                    "ProbeType": res["ProbeType"],
                    "PCA": res["PCA"],
                    "ValAccuracies": acc
                })

    df = pd.DataFrame(flat_records)
    df = df[df["ProbeType"] == probe_filter]
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)

    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())

    # Global style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "axes.facecolor": "#e8ecf0",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.2,
    })

    # Compute KDE-based global y-limits for this probe type
    y_vals = df["ValAccuracies"].values
    if len(y_vals) > 1:
        kde = gaussian_kde(y_vals)
        x_range = np.linspace(y_vals.min(), y_vals.max(), 500)
        density = kde(x_range)
        cutoff_vals = x_range[density > max(density) * 0.01]
        y_min = cutoff_vals.min() if len(cutoff_vals) else y_vals.min()
        y_max = cutoff_vals.max() if len(cutoff_vals) else y_vals.max()
    else:
        y_min, y_max = y_vals.min(), y_vals.max()

    padding = (y_max - y_min) * 0.25 if y_max > y_min else 0.05
    y_min_final = max(0.0, y_min - padding)
    y_max_final = min(1.2, y_max + padding)

    # FacetGrid with one subplot per Beta
    g = sns.FacetGrid(
        df, col="Beta", col_wrap=4,
        height=3.0, aspect=1.2,
        sharey=True,  # consistent scale across all beta subplots
        col_order=beta_order
    )

    def draw_pca_grouped_boxplot(data, **kwargs):
        ax = plt.gca()
        ax.set_facecolor("#e8ecf0")

        sns.violinplot(
            data=data, x="PCA", y="ValAccuracies",
            ax=ax, order=pca_order,
            inner=None, linewidth=0,
            color="#a0aab8", saturation=0.3
        )

        sns.boxplot(
            data=data, x="PCA", y="ValAccuracies",
            width=0.3, ax=ax, order=pca_order,
            color="white",
            fliersize=1.5, linewidth=0.7,
            boxprops={'facecolor': 'white', 'edgecolor': '#333', 'zorder': 2},
            whiskerprops={'linewidth': 0.7},
            capprops={'linewidth': 0.7},
            medianprops={'color': 'black', 'linewidth': 1}
        )

        medians = data.groupby("PCA", observed=True)["ValAccuracies"].median().reindex(pca_order)
        x_vals = list(range(len(pca_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(pca_order)
        ax.set_xlabel("PCA Components", fontsize=10)
        ax.set_ylabel("Validation Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(y_min_final, y_max_final)

    g.map_dataframe(draw_pca_grouped_boxplot)
    g.set_titles(col_template=r"$\beta$ = {col_name}", size=10)
    plt.suptitle(f"Validation Accuracy by PCA Dimension\n({probe_filter} Probes)", fontsize=14, y=1.05)

    plt.tight_layout()
    plot_path = Path(f"{project_path}/pca_over_beta_faceted_by_beta.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PCA-vs-Beta plot to: {plot_path}")
    plt.show()
    plt.close()




def plot_pca_comparison_by_beta_grid(json_path, project_path="test_outputs"):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten records
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append({
                    "Model": model["Model"],
                    "Beta": model["Beta"],
                    "SourceTimestep": model["SourceTimestep"],
                    "TargetTimestep": model["TargetTimestep"],
                    "ProbeType": res["ProbeType"],
                    "PCA": res["PCA"],
                    "ValAccuracies": acc
                })

    df = pd.DataFrame(flat_records)
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)

    probe_order = ["Linear", "Two-Layer"]
    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())

    set_plot_style()

    # Calculate y-axis limits per ProbeType
    y_limits = {
        probe: (
            df[df["ProbeType"] == probe]["ValAccuracies"].min(),
            df[df["ProbeType"] == probe]["ValAccuracies"].max()
        )
        for probe in probe_order
    }

    # Use sharey=False for independent scaling
    g = sns.FacetGrid(
        df,
        row="ProbeType", col="Beta",
        height=2.8, aspect=1.2,
        sharey=False,
        row_order=probe_order,
        col_order=beta_order,
        margin_titles=True
    )

    def draw_pca_vs_accuracy_boxplot(data, **kwargs):
        ax = plt.gca()
        probe_type = data["ProbeType"].iloc[0]
        ax.set_facecolor("#e8ecf0")

        sns.violinplot(
            data=data, x="PCA", y="ValAccuracies",
            ax=ax, order=pca_order,
            inner=None, linewidth=0, color="#a0aab8", saturation=0.3
        )

        sns.boxplot(
            data=data, x="PCA", y="ValAccuracies",
            width=0.3, ax=ax, order=pca_order,
            color="white",
            fliersize=1.5, linewidth=0.7,
            boxprops={'facecolor': 'white', 'edgecolor': '#333', 'zorder': 2},
            whiskerprops={'linewidth': 0.7},
            capprops={'linewidth': 0.7},
            medianprops={'color': 'black', 'linewidth': 1}
        )

        medians = data.groupby("PCA", observed=True)["ValAccuracies"].median().reindex(pca_order)
        x_vals = list(range(len(pca_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(pca_order)
        ax.set_xlabel("PCA Components", fontsize=10)
        ax.set_ylabel("Validation Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Set y-limit for this row
        y_min, y_max = y_limits[probe_type]
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
        ax.set_ylim(y_min - padding, y_max + padding)

    g.map_dataframe(draw_pca_vs_accuracy_boxplot)
    g.set_titles(col_template=r"$\beta$ = {col_name}", row_template="", size=10)

    # Row titles
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation"
    }

    g.fig.subplots_adjust(top=0.92, hspace=0.35)

    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = 0.1 if i == len(probe_order) - 1 else 0.035

        g.fig.text(
            center_x, bottom_y - y_offset,
            row_labels[probe],
            fontsize=11, fontweight="bold", ha="center", va="top"
        )

    plt.suptitle("Validation Accuracy across PCA Dimensions per Beta Value", fontsize=14, y=1.05)
    plot_path = Path(f"{project_path}/pca_vs_beta_combined_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved combined PCA/Beta grid plot to: {plot_path}")
    plt.show()
    plt.close()




def plot_accuracy_grid_from_json(json_path, probe_filter="Linear"):
    """
    Loads structured PCA evaluation results, flattens them, and plots
    validation accuracy over epochs as a grid (Faceted by PCA x Beta).
    """
    # ---------------------------
    # 1. Load JSON data
    # ---------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---------------------------
    # 2. Flatten records
    # ---------------------------
    flat_records = []
    for model in data:
        for res in model["Results"]:
            if res["ProbeType"] != probe_filter:
                continue
            for epoch_idx, acc in enumerate(res["ValAccuracies"]):
                flat_records.append({
                    "Model": model["Model"],
                    "Beta": float(model["Beta"]),
                    "SourceTimestep": model["SourceTimestep"],
                    "TargetTimestep": model["TargetTimestep"],
                    "ProbeType": res["ProbeType"],
                    "PCA": int(res["PCA"]),
                    "Epoch": epoch_idx,
                    "ValAccuracy": acc
                })

    df = pd.DataFrame(flat_records)

    if df.empty:
        print("[WARN] No data found for probe type:", probe_filter)
        return

    # ---------------------------
    # 3. Normalize y-axis per row
    # ---------------------------
    # Get max accuracy per Beta (row) for setting y-lims
    beta_max = df.groupby("Beta")["ValAccuracy"].max().to_dict()

    # Function to use for y-limits per Facet
    def adjust_ylim(data, color, **kwargs):
        beta = data["Beta"].iloc[0]
        ymax = beta_max[beta]
        sns.lineplot(data=data, x="Epoch", y="ValAccuracy", color=color, **kwargs)
        plt.ylim(0, ymax * 1.05)

    # ---------------------------
    # 4. Plot grid
    # ---------------------------
    sns.set(style="whitegrid", font_scale=0.8)

    g = sns.FacetGrid(
        df,
        row="Beta",
        col="PCA",
        margin_titles=True,
        sharey=False,
        height=2.5,
        aspect=1.5
    )
    g.map_dataframe(adjust_ylim)

    g.set_axis_labels("Epoch", "Validation Accuracy")
    g.set_titles(row_template="β = {row_name}", col_template="PCA = {col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Validation Accuracy for {probe_filter} Probes Across PCA Components", fontsize=14)

    plt.show()





######################################################################
#                   Collect Latents                                  #
######################################################################
def collect_latents_from_dataloader(
    data_path,
    batch_size,
    source_timestep,
    target_timestep,
    group,
    results_root,
    project_path,
    model_name,
    beta_vae_module,
    device=None,
    max_samples=50000
):
    # Set device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    base_results_dir = Path(results_root) / project_path / model_name / timestamp
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {base_results_dir}")

    # Collect latents
    with torch.no_grad():
        all_latents, all_labels, curr_samples = [], [], 0
        print("\n--- Collecting bottleneck latents ---")
        for batch in tqdm(dataloader, desc="Collecting latents"):
            if curr_samples >= max_samples:
                break

            source_latents = batch[f'latents_{source_timestep:.2f}'].to(device, non_blocking=True)
            encoded = beta_vae_module.model.encode(source_latents)
            latents = encoded['latent_dist'].mode()
            all_latents.append(latents.detach().cpu().numpy())

            if 'label' in batch:
                all_labels.append(batch['label'].detach().cpu().numpy())
            curr_samples += latents.shape[0]

    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    print(f"[INFO] Collected latents shape: {all_latents.shape}")
    print(f"[INFO] Collected labels shape: {all_labels.shape if all_labels is not None else 'N/A'}")

    return all_latents, all_labels




######################################################################
#                   Full PCA Evaluation Pipeline                      #
######################################################################

from pathlib import Path
import json
import torch
import gc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def run_pca_over_beta(
    source_timestep=0.20,
    target_timestep=1.00,
    beta=1.0,
    dataset_name='imagenet256-dataset',
    group="validation",
    checkpoint=None,
    data_path=None,           # Should be Path or str
    project_path=None,        # Should be Path object!
    model_name=None,
    num_components=20,
    max_data_samples=50000,
    max_umap_samples=20000,
    pca_latent_numbers=[2, 3, 5, 7, 10, 20],
    batch_size=32,
    epochs=500,
    patience=10,
    lr=1e-4,
    device=None,
    results_root=None         # Not used for saving in this function
):
    # Ensure correct path types up front
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if isinstance(project_path, str):
        project_path = Path(project_path)
    if results_root is not None and isinstance(results_root, str):
        results_root = Path(results_root)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(checkpoint, map_location='cpu')
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

    all_latents, all_labels = collect_latents_from_dataloader(
        data_path,
        batch_size,
        source_timestep,
        target_timestep,
        group,
        results_root,
        project_path,
        model_name,
        beta_vae_module,
        device=device,
        max_samples=max_data_samples
    )

    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)

    unique_labels = np.unique(all_labels)
    label_map = {int(lbl): int(idx) for idx, lbl in enumerate(sorted(unique_labels))}
    inverse_label_map = {v: k for k, v in label_map.items()}
    all_labels_mapped = np.vectorize(label_map.get)(all_labels)
    num_classes = len(unique_labels)

    # ---- PATHS: This is now robust and readable! ----
    # Everything for this run/model goes in here:
    base_results_dir = project_path / f"pca_evaluation_{model_name}"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Save label maps
    with (base_results_dir / "label_map.json").open("w") as f:
        json.dump(label_map, f, indent=2)
    with (base_results_dir / "inverse_label_map.json").open("w") as f:
        json.dump(inverse_label_map, f, indent=2)

    structured_results = {
        "Model": model_name,
        "Beta": beta,
        "Source_TS": source_timestep,
        "Target_TS": target_timestep,
        "Results": []
    }

    for pca_num in pca_latent_numbers:
        if pca_num > num_components:
            continue

        print(f"\n--- Evaluating PCA with {pca_num} components ---")
        pca_latents_subset = pca_latents[:, :pca_num]
        result_dir = base_results_dir / f"pca_{pca_num}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            pca_latents_subset, all_labels_mapped,
            test_size=0.2, random_state=42, stratify=all_labels_mapped
        )
        train_loader = DataLoader(PCADataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(PCADataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        print("Training Linear Probe...")
        linear_probe = LinearProbe(hidden_size=pca_num, num_classes=num_classes)
        df_linear = train_linear_probe(
            linear_probe=linear_probe,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            patience=patience,
            lr=lr,
            output_csv=result_dir / "linear_probe.csv",
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            beta_value=beta,
            model_name=model_name
        )
        structured_results["Results"].append({
            "ProbeType": "Linear",
            "PCA": pca_num,
            "ValAccuracies": df_linear["Val_Accuracy"].tolist()
        })

        print("Training Two-Layer Probe...")
        two_layer_probe = TwoLayerProbe(input_dim=pca_num, hidden_dim=128, num_classes=num_classes)
        df_two = train_linear_probe(
            linear_probe=two_layer_probe,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            patience=patience,
            lr=lr,
            output_csv=result_dir / "two_layer_probe.csv",
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            beta_value=beta,
            model_name=model_name + "_TwoLayer"
        )
        structured_results["Results"].append({
            "ProbeType": "Two-Layer",
            "PCA": pca_num,
            "ValAccuracies": df_two["Val_Accuracy"].tolist()
        })

        # Save PCA outputs
        np.save(result_dir / "pca_latents.npy", pca_latents_subset)
        np.save(result_dir / "pca_labels_mapped.npy", y_val)
        np.save(result_dir / "pca_labels_original.npy", all_labels)

    # Save all structured results at the end
    with (base_results_dir / "structured_results.json").open("w") as f:
        json.dump(structured_results, f, indent=2)

    return structured_results




######################################################################
#                   Full PCA Evaluation Pipeline                      #
######################################################################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def run_pca_for_kmeans_and_plot_scatter_grid(
    source_timestep=0.20,
    target_timestep=1.00,
    beta=1.0,
    dataset_name='imagenet256-dataset',
    group="validation",
    checkpoint=None,
    data_path=None,
    project_path=None,
    model_name=None,
    num_components=20,
    max_data_samples=50000,
    pca_latent_numbers=[2, 3, 5, 7, 10, 20],
    k_values=[2, 3, 5, 10, 15, 20],
    batch_size=32,
    device=None,
    results_root="results"
):
    """ Run PCA on β-VAE latents and plot K-Means clustering scatter grid."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()

    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(checkpoint, map_location="cpu")
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

    all_latents, all_labels = collect_latents_from_dataloader(
        data_path,
        batch_size,
        source_timestep,
        target_timestep,
        group,
        results_root,
        project_path,
        model_name,
        beta_vae_module,
        device=device,
        max_samples=max_data_samples
    )

    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)

    scatter_data = []

    # Create a DataFrame to hold the scatter plot data
    for pca_num in pca_latent_numbers:
        if pca_num < 2 or pca_num > num_components:
            continue

        pca_subset = pca_latents[:, :pca_num]

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            clusters = kmeans.fit_predict(pca_subset)

            for i in range(len(pca_subset)):
                scatter_data.append({
                    "x": pca_subset[i, 0],
                    "y": pca_subset[i, 1],
                    "PCA": pca_num,
                    "K": k,
                    "Cluster": clusters[i]
                })

    # After collecting data for scatter
    df_scatter = pd.DataFrame(scatter_data)
    plot_pca_kmeans_scatter_grid(df_scatter, project_path=project_path or "test_outputs")





if __name__ == "__main__":
    #####################################
    # Shared Parameters
    #####################################
    dataset_name        = 'imagenet256-dataset-T000006'
    group               = "validation"
    num_components      = 5 #50
    max_data_samples    = 50 # 100000
    batch_size          = 64
    data_path           = './dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5'
    results_path        = './results/PCA_BetaVAE_Eval'


    n_neighbors         = 50
    min_dist            = 0.1
    max_umap_samples    = 50 #25000
    random_state        = 42 # state of life
    epochs              = 500
    patience            = 10
    lr                  = 1e-4

    pca_latent_numbers  = [4] # [2, 5, 9, 15, 20, 30, 50]
    k_means_n_values    = [4] #[2, 5, 9, 15, 20, 30, 50]

    #####################################
    # Device + Seed Setup
    #####################################
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()


    # --------------------------------------
    # Set base results directory with date
    # --------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_root = Path(results_path) / timestamp
    experiment_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {experiment_root}")

    # --------------------------------------
    # Model Configurations
    # --------------------------------------

    model_configs_v0 = [
        # mixed beta = {1e-4, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0}
        {"name": "Beta02x10x_1e4b", "beta": 1e-4, "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_01b",  "beta": 0.1,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_05b",  "beta": 0.5,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_1b",   "beta": 1.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_2b",   "beta": 2.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-2.0b/V2/2025-07-06/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_3b",   "beta": 3.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt'},
        {"name": "Beta02x10x_5b",   "beta": 5.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'},
    ]
    model_configs_v1 = [
        # mixed beta = {1e-4, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0}
        {"name": "Beta05x10x_01b", "beta": 0.1,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-08-04/manual/V2/2025-08-04/100001/checkpoints/last.ckpt'},
        {"name": "Beta05x10x_05b", "beta": 0.5,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-31/101646/checkpoints/last.ckpt'},
        {"name": "Beta05x10x_1b",  "beta": 1.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'},
        {"name": "Beta05x10x_2b",  "beta": 2.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-2b/2025-06-30-1435/manual/V2/2025-07-31/101646/checkpoints/last.ckpt'},
        {"name": "Beta05x10x_5b",  "beta": 5.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'},
    ]

    all_config_groups = [
        {"group_name": "All_BetaVAE_0.2x1.0x",
         "configs": model_configs_v0},
        {"group_name": "All_BetaVAE_0.5x1.0x",
         "configs": model_configs_v1},
    ]

    # --------------------------------------
    # Main loop over all config groups
    # --------------------------------------
    for config_group in all_config_groups:
        group_name = config_group["group_name"]
        configs = config_group["configs"]
        model_results = []

        print(f"\n=== Running group: {group_name} ===")

        for config in configs:
            beta        = config["beta"]
            checkpoint  = config["ckpt"]
            source_ts   = config["source_ts"]
            target_ts   = config["target_ts"]
            model_tag   = config["name"]
            model_name  = f"{model_tag}_{dataset_name}"

            # Use experiment_root here
            model_path = experiment_root / f"PCA_Quantitative_{group_name}" / model_tag
            model_path.mkdir(parents=True, exist_ok=True)

            print(f"\n[INFO] Running model: {model_tag} (β={beta}, source={source_ts:.2f}, target={target_ts:.2f})")

            structured_results = run_pca_over_beta(
                source_timestep=source_ts,
                target_timestep=target_ts,
                beta=beta,
                pca_latent_numbers=pca_latent_numbers,
                dataset_name=dataset_name,
                group=group,
                checkpoint=checkpoint,
                data_path=Path(data_path),
                project_path=model_path,
                model_name=model_name,
                num_components=num_components,
                max_data_samples=max_data_samples,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lr=lr,
                results_root=experiment_root,
            )

            assert isinstance(structured_results.get("Results", None), list), f"Results missing or malformed in {model_tag}"

            structured_results.update({
                "Model": model_tag,
                "Beta": beta,
                "SourceTimestep": source_ts,
                "TargetTimestep": target_ts,
            })

            model_results.append(structured_results)


        # Save JSON
        group_path = experiment_root / f"{group_name}_PCA"
        group_path.mkdir(exist_ok=True, parents=True)
        out_path = group_path / "group_probe_results.json"
        with out_path.open("w") as f:
            json.dump(model_results, f, indent=2)
        print(f"[INFO] Saved results to: {out_path}")

        # Plotting
        print(f"==" * 50)
        print(f"[INFO] Generating plots...")
        plot_accuracy_grid_from_json(
            json_path=out_path,
            probe_filter="Linear"
        )

        plot_probe_val_across_pca(
            json_path=out_path,
            project_path=group_path,
            probe_filter="Linear"
        )
        plot_probe_comparison_grid(
            json_path=out_path,
            project_path=group_path,
        )
        plot_pca_across_betas(
            json_path=out_path,
            project_path=group_path,
            probe_filter="Linear"
        )
        plot_pca_comparison_by_beta_grid(
            json_path=out_path,
            project_path=group_path,
        )
        print(f"[INFO] Completed group: {group_name}\n")
        print(f"==" * 50)s

    # --------------------------------------
    # K-Means PCA Scatter Grid (Fixed)
    # --------------------------------------
    for config_group in all_config_groups:
        group_name = config_group["group_name"]
        configs = config_group["configs"]

        print(f"\n=== Running K-Means visualisation for group: {group_name} ===")

        for config in configs:
            beta        = config["beta"]
            checkpoint  = config["ckpt"]
            source_ts   = config["source_ts"]
            target_ts   = config["target_ts"]
            model_tag   = config["name"]
            model_name  = f"{model_tag}_{dataset_name}"

            # use experiment_root
            project_path = experiment_root / f"PCA_Quantitative_{group_name}" / model_tag
            kmeans_output_path = project_path / f"kmeans_pca_{model_tag}"
            kmeans_output_path.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] KMeans for model: {model_tag} (β={beta})")

            run_pca_for_kmeans_and_plot_scatter_grid(
                source_timestep=source_ts,
                target_timestep=target_ts,
                beta=beta,
                dataset_name=dataset_name,
                group=group,
                checkpoint=checkpoint,
                data_path=Path(data_path),
                project_path=project_path,
                model_name=model_name,
                num_components=num_components,
                max_data_samples=max_data_samples,
                pca_latent_numbers=pca_latent_numbers,
                k_values=k_means_n_values,
                batch_size=batch_size,
                results_root=kmeans_output_path,
            )



    # CUDA_VISIBLE_DEVICES=2 python ...







    # -------------------------------------------------------
    # B: Reconstruction + Varying ß-Parameter
    # All models under Denoising Objective
    # -------------------------------------------------------



    # # All models under Reconstruction Objective
    # # -------------------------------------------------------
    # model_configs_v0 = [
    #     # beta: 0.1
    #     {"name": "Beta05x05x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt' },  # Open (Baseline)
    #     {"name": "Beta02x02x_01b",  "beta": 0.1,  "source_ts": 0.20, "target_ts": 0.20, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt' },  # Open
    #     {"name": "Beta00x00x_01b",  "beta": 0.1,  "source_ts": 0.00, "target_ts": 0.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt' },  # Open
    #     # beta: 1.0
    #     {"name": "Beta05x05x_1b",  "beta": 1.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt' },  # Open
    #     # beta: 5.0
    #      {"name": "Beta05x05x_5b",  "beta": 5.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt' },  # Open
    # ]


    # # All models with b:0.1
    # model_configs_v2 = [
    #     # beta: 0.1
    #     {"name": "Beta05x05x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt' },  # Open (Baseline)
    #     {"name": "Beta05x10x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-02/101646/checkpoints/last.ckpt'},
    #     {"name": "Beta04x10x_01b",  "beta": 0.1,  "source_ts": 0.40, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.40x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},
    #     {"name": "Beta03x10x_01b",  "beta": 0.1,  "source_ts": 0.30, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.30x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt'},
    #     {"name": "Beta02x10x_01b",  "beta": 0.1,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt'},
    #     {"name": "Beta00x10x_01b",  "beta": 0.1,  "source_ts": 0.00, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-1.00x-0.1b/2025-06-18/29852/V0-eV2/2025-06-24/29852/checkpoints/last.ckpt'},
    # ]

    # # All models with b:1.0
    # model_configs_v3 = [
    #     # beta: 1.0
    #     {"name": "Beta05x05x_1b",   "beta": 1.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt'}, # Open (Baseline)
    #     {"name": "Beta05x10x_1b",   "beta": 1.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt'},
    #     {"name": "Beta02x10x_1b",   "beta": 1.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt'},
    # ]

    # # All models with b:5.0
    # model_configs_v4 = [
    #     # beta: 5.0
    #     {"name": "Beta05x05x_5b",  "beta": 5.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt'}, # Open (Baseline)
    #     {"name": "Beta05x10x_5b",  "beta": 5.0,  "source_ts": 0.50, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt'},
    #     {"name": "Beta02x10x_5b",  "beta": 5.0,  "source_ts": 0.20, "target_ts": 1.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt'},
    # ]

    # # All models with fixed beta:0.1
    # model_configs_v5 = [
    #     # Self-reconstruction tasks
    #     {"name": "Beta05x05x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt' },  # Open (Baseline)
    #     {"name": "Beta02x02x_01b",  "beta": 0.1,  "source_ts": 0.20, "target_ts": 0.20, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt' },
    #     {"name": "Beta00x00x_01b",  "beta": 0.1,  "source_ts": 0.00, "target_ts": 0.00, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt' },
    # ]

    # # All baseline models with different betas
    # model_configs_v6 = [
    #     # comparison of all baseline models with source: 0.5 -> target 0.5
    #     {"name": "Beta05x05x_01b",  "beta": 0.1,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt' },  # Open (Baseline)
    #     {"name": "Beta05x05x_1b",   "beta": 1.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt'}, # Open (Baseline)
    #     {"name": "Beta05x05x_5b",  "beta": 5.0,  "source_ts": 0.50, "target_ts": 0.50, "ckpt": './logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt'}, # Open (Baseline)
    # ]
