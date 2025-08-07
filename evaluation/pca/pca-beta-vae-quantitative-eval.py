import os
import sys
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt

import json

import math
import gc
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torch.nn as nn
import torch.optim as optim
from lightning import seed_everything


import seaborn as sns
import pandas as pd
import numpy as np
import umap
from tqdm import tqdm



from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE



# helper Jutils imports
from jutils import freeze


# Setup project root for import resolution
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
)
sys.path.append(project_root)

from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule


torch.set_float32_matmul_precision("high")


##########################################################################
#                   Find Directions with PCA                             #
##########################################################################


@torch.no_grad()
def find_pca_directions(
    module, dataloader, source_timestep=0.5, num_components=10, device=None
):
    device = device or (module.device if hasattr(module, "device") else "cpu")
    print(f"[INFO] Collecting latents for PCA on device: {device}")

    all_latents = []
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        source_latents = batch[f"latents_{source_timestep:.2f}"].to(
            device, non_blocking=True
        )
        encoded = module.model.encode(source_latents)
        latents = encoded["latent_dist"].mode()
        all_latents.append(latents.detach().cpu().numpy())

    combined_latents = np.vstack(all_latents)
    print(
        f"[INFO] Collected {combined_latents.shape[0]} latent vectors of dim {combined_latents.shape[1]}."
    )

    # Sorted by vairance (highest --> lowest)
    pca = PCA(n_components=num_components)
    pca.fit(combined_latents)

    print(f"[INFO] PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"[INFO] Total explained: {np.sum(pca.explained_variance_ratio_):.2f}")

    return pca.components_, pca.explained_variance_ratio_


############################################################################
#                   Visualise most Important PCA Vectors                   #
############################################################################


def plot_pca_2d_projection(
    pca_latents, labels=None, save_path=None, title="PCA 2D Projection"
):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(
            pca_latents[:, 0],
            pca_latents[:, 1],  # first and second PCA components
            c=labels,
            cmap="tab20",
            s=5,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Class label")
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
        ax.hist(pca_latents[:, i], bins=num_bins, color="cornflowerblue", alpha=0.7)
        ax.set_title(f"PCA Component {i + 1}")
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
        item = {"pca": torch.tensor(self.pca_latents[idx], dtype=torch.float32)}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_pca_dataloader(
    pca_latents, labels=None, batch_size=32, shuffle=True, num_workers=4
):
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
    return DataLoader(
        pca_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


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
            nn.Linear(hidden_dim, num_classes),
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
    label_key="label",
    latent_key="pca",
    device="cuda",
    epochs=500,
    patience=10,
    lr=1e-4,
    output_csv="linear_probe_metrics.csv",
    beta_value=1e-4,  # default low beta value for β-VAE
    model_name="",
    output_dim=90,
):
    linear_probe = linear_probe.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        linear_probe.parameters(), lr=lr
    )  # use AdamW for better generalization

    history = []
    best_val_acc = -float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        linear_probe.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}", leave=False):
            pca_vectors = batch[latent_key].to(device)
            labels = batch[label_key].to(device).view(-1)

            # Check if labels are within valid range
            if (labels < 0).any() or (labels >= linear_probe.get_output_dim()).any():
                print(
                    f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}"
                )
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
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}", leave=False):
                pca_vectors = batch[latent_key].to(device)
                labels = batch[label_key].to(device)

                # Check if labels are within valid range
                if (labels < 0).any() or (labels >= linear_probe.output_dim).any():
                    print(
                        f"[WARNING] Skipping batch due to invalid labels: {labels.cpu().numpy()}"
                    )
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
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": train_loss,
                "Train_Accuracy": train_acc,
                "Val_Loss": val_loss,
                "Val_Accuracy": val_acc,
                "Precision": precision,
                "Recall": recall,
                "Beta": beta_value,
                "Model": model_name,
                "Source_Timestep": source_timestep,
                "Target_Timestep": target_timestep,
            }
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] Stopped at epoch {epoch + 1}")
                break

        print(
            f"[Epoch {epoch + 1}] Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
        )

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
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "axes.facecolor": "#f5f5f5",
            "axes.edgecolor": "#333333",
            "grid.linestyle": ":",
            "grid.color": "grey",
            "axes.grid": True,
        }
    )
    sns.set(style="whitegrid")
    sns.set_palette("Set2")  # Ensures consistent coloring for all plots


def plot_validation_curve(
    df_metrics, source_timestep, target_timestep, beta, save_path=None
):
    """
    Plot a polished validation accuracy curve using seaborn and consistent style.
    """
    set_plot_style()

    epochs = df_metrics["Epoch"]
    val_acc = df_metrics["Val_Accuracy"]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Use the first color from the Set2 palette
    color = sns.color_palette("Set2", n_colors=1)[0]

    ax.plot(
        epochs,
        val_acc,
        marker="o",
        color=color,
        linewidth=2.5,
        markersize=7,
        label="Validation Accuracy",
    )

    ax.text(
        epochs.values[-1],
        val_acc.values[-1] + 0.01,
        f"{val_acc.values[-1] * 100:.1f}%",
        ha="center",
        fontsize=11,
        color=color,
    )

    ax.set_title(
        rf"Validation Accuracy - β-VAE   (source={source_timestep:.2f} → target={target_timestep:.2f},  β={beta})",
        fontsize=16,
        pad=20,
    )

    ymax = math.ceil((val_acc.max() + 0.05) * 20) / 20
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(
        fontsize=12, loc="upper right", title="Model", title_fontsize=13, frameon=False
    )

    plt.tight_layout()

    if save_path is None:
        save_path = f"validation_curve_{source_timestep:.2f}_{target_timestep:.2f}_beta{beta}.png"
        print(f"[INFO] No save path provided, using default: {save_path}")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"[INFO] Validation curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_combined_validation_curve(
    df_combined, save_path, source_timestep, target_timestep, beta
):
    """
    Plot a combined validation accuracy curve for multiple probes.
    """
    set_plot_style()

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=df_combined["Probe"].nunique())

    ax = sns.lineplot(
        data=df_combined,
        x="Epoch",
        y="Val_Accuracy",
        hue="Probe",
        marker="o",
        palette=palette,
    )

    for probe, sub_df in df_combined.groupby("Probe"):
        last_row = sub_df.iloc[-1]
        ax.text(
            last_row["Epoch"] + 0.2,
            last_row["Val_Accuracy"],
            f"{last_row['Val_Accuracy'] * 100:.2f}%",
            fontsize=11,
            color=ax.get_lines()[
                list(df_combined["Probe"].unique()).index(probe)
            ].get_color(),
            weight="bold",
        )

    ymax = math.ceil((df_combined["Val_Accuracy"].max() + 0.05) * 20) / 20
    plt.ylim(0, ymax)

    plt.title(
        f"Validation Accuracy Comparison (β={beta}, source={source_timestep:.2f} → target={target_timestep:.2f})",
        fontsize=16,
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=14)
    plt.legend(title="Probe Type", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Combined validation curve saved to: {save_path}")
    plt.show()
    plt.close()


############################################################################
#                       Plot UMAP 2D Cluster Plot                          #
############################################################################
def plot_umap_pca(
    pca_latents,  # PCA-projected latents, shape (N, D)
    labels=None,
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    max_data_samples=100000,
    save_to_path=None,
    title="UMAP projection of PCA latents",
    figsize=(10, 8),
    random_state=42,  # state of life s
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
    if pca_latents.shape[0] > max_data_samples:
        pca_latents = pca_latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None
        print(
            f"[INFO] PCA shape reduced to {pca_latents.shape[0]} samples for plotting."
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(pca_latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=5, alpha=0.7
        )
        plt.colorbar(scatter, label="Class label")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] UMAP plot saved to: {save_to_path}")
    plt.show()


def plot_pca_2d(
    latents,
    labels=None,
    max_data_samples=100000,
    save_to_path=None,
    title="2D PCA Projection",
    figsize=(10, 8),
):
    if latents.shape[0] > max_data_samples:
        latents = latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=5, alpha=0.7
        )
        plt.colorbar(scatter, label="Class label")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] PCA plot saved to: {save_to_path}")
    plt.show()


def plot_tsne_2d(
    latents,
    labels=None,
    max_data_samples=10000,
    perplexity=30,
    save_to_path=None,
    title="t-SNE Projection",
    figsize=(10, 8),
):
    if latents.shape[0] > max_data_samples:
        latents = latents[:max_data_samples]
        labels = labels[:max_data_samples] if labels is not None else None

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    reduced = tsne.fit_transform(latents)

    plt.figure(figsize=figsize)
    if labels is not None:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=5, alpha=0.7
        )
        plt.colorbar(scatter, label="Class label")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    if save_to_path:
        plt.savefig(save_to_path, bbox_inches="tight", dpi=300)
        print(f"[INFO] t-SNE plot saved to: {save_to_path}")
    plt.show()


def plot_kmeans_grid(
    pca_latents,
    k_values=[2, 3, 5, 10],
    max_data_samples=100000,
    figsize_per_plot=(5, 5),
    save_to_path=None,
):
    """
    Plot k-Means clustering results over PCA-reduced latents (assumed to be already PCA'd).

    Args:
        pca_latents: Already PCA-reduced latents (e.g., shape [N, D])
        k_values: List of k values to try for k-means
        max_data_samples: Subsample limit for performance
        figsize_per_plot: Size per subplot
        save_to_path: If provided, saves the resulting figure
    """
    if pca_latents.shape[0] > max_data_samples:
        pca_latents = pca_latents[:max_data_samples]

    assert pca_latents.shape[1] >= 2, "Need at least 2 PCA components"

    reduced = pca_latents[:, :2]  # Only use first two PCA components

    n_rows = 2
    n_cols = int(np.ceil(len(k_values) / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42)
        preds = kmeans.fit_predict(pca_latents)  # Cluster in full PCA space

        ax = axes[idx]
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1], c=preds, cmap="tab20", s=5, alpha=0.7
        )
        ax.set_title(f"k-Means Clustering (k={k})")
        ax.axis("off")

    # Hide any extra unused subplots
    for ax in axes[len(k_values) :]:
        ax.axis("off")

    plt.suptitle("k-Means Cluster Projections (PCA-reduced)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_to_path:
        plt.savefig(save_to_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] k-Means grid plot saved to: {save_to_path}")

    plt.show()


def plot_umap_grid(
    model_results,
    group_name,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    save_path=None,
    max_data_samples=50000,
    random_state=None,
):
    """
    Plot UMAP plots for multiple models in a grid.

    Args:
        model_results (list of dict): Each dict has 'latents', 'labels', 'name'.
        group_name (str): Name of the group (for title).
        save_path (str, optional): If provided, saves the grid plot.
        max_data_samples (int): Maximum samples to plot per model.
    """
    num_models = len(model_results)
    n_cols = 3  # You can adjust
    n_rows = int(np.ceil(num_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )

    for idx, model_info in enumerate(model_results):
        row = idx // n_cols
        col = idx % n_cols

        latents = model_info["latents"]
        labels = model_info["labels"]
        name = model_info["name"]

        if latents.shape[0] > max_data_samples:
            latents = latents[:max_data_samples]
            labels = labels[:max_data_samples] if labels is not None else None

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(latents)

        ax = axes[row][col]
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=3, alpha=0.7
            )
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


######################################################################
#                   Full PCA Evaluation Pipeline                      #
######################################################################
def run_pca_eval(
    source_timestep=0.20,
    target_timestep=1.00,
    beta=1.0,
    dataset_name="imagenet256-dataset",
    group="validation",
    checkpoint=None,
    data_path=None,
    project_name=None,
    model_name=None,
    num_components=5,
    max_data_samples=50000,
    max_umap_samples=20000,
    batch_size=32,
    epochs=500,
    patience=10,
    lr=1e-4,
    device=None,
    results_root="results",
):
    """
    Full pipeline to run PCA on bottleneck latents, remap labels to contiguous range,
    train a linear probe, save metrics, components, plots, and label mappings.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    gc.collect()

    # Load VAE model
    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)  # Freeze the VAE model

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
    dataloader = (
        data.val_dataloader() if group == "validation" else data.test_dataloader()
    )

    # Results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            if curr_samples >= max_data_samples:
                break

            # Encode to Z-space latents
            source_latents = batch[f"latents_{source_timestep:.2f}"].to(
                device, non_blocking=True
            )
            encoded = beta_vae_module.model.encode(source_latents)
            latents = encoded["latent_dist"].mode()
            all_latents.append(latents.detach().cpu().numpy())

            # Collect labels if available
            if "label" in batch:
                all_labels.append(batch["label"].detach().cpu().numpy())
            curr_samples += latents.shape[0]

    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    print(f"[INFO] Collected latents shape: {all_latents.shape}")
    print(
        f"[INFO] Collected labels shape: {all_labels.shape if all_labels is not None else 'N/A'}"
    )

    ########################################################################
    #                       Fit PCA and project                           #
    ########################################################################
    print(f"[INFO] Fitting PCA with {num_components} components ...")
    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)
    print(f"[INFO] Explained variance: {pca.explained_variance_ratio_}")

    ########################################################################
    #                  Robust label mapping & save                        #
    ########################################################################
    unique_labels = np.unique(all_labels)
    label_map = {
        int(original): int(idx) for idx, original in enumerate(sorted(unique_labels))
    }
    inverse_label_map = {int(idx): int(original) for original, idx in label_map.items()}

    all_labels_mapped = np.vectorize(label_map.get)(all_labels)
    num_classes = len(unique_labels)
    print(
        f"[INFO] Found {num_classes} unique classes. Labels remapped to 0–{num_classes - 1}."
    )

    # Save mappings
    with open(base_results_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(base_results_dir / "inverse_label_map.json", "w") as f:
        json.dump(inverse_label_map, f, indent=2)
    print("[INFO] Saved label_map and inverse_label_map.")

    ########################################################################
    #                  Split into train and val sets                      #
    ########################################################################
    X_train, X_val, y_train, y_val = train_test_split(
        pca_latents,
        all_labels_mapped,
        test_size=0.2,
        random_state=42,
        stratify=all_labels_mapped,
    )
    train_loader = DataLoader(
        PCADataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        PCADataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4
    )

    for loader in [train_loader, val_loader]:
        for batch in loader:
            assert "pca" in batch, "Expected 'pca' key in batch for LinearProbe"

    print(
        f"[INFO] Train set size: {len(train_loader.dataset)}, Val set size: {len(val_loader.dataset)}"
    )

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
        epochs=epochs,
        patience=patience,
        lr=lr,
        output_csv=str(output_csv),
        beta_value=beta,
        model_name=model_name,
    )

    print(f"[INFO] Linear probe metrics saved to: {output_csv}")
    plot_path = base_results_dir / "val_accuracy_curve.png"
    plot_validation_curve(
        df_metrics,
        save_path=plot_path,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        beta=beta,
    )
    print("[INFO] Linear probe training completed.")

    ########################################################################
    #                        Train Multi-Layer probe                       #
    ########################################################################
    print("\n--- Training Multi-Layer Probe ---")
    multi_layer_probe = TwoLayerProbe(
        input_dim=num_components, hidden_dim=128, num_classes=num_classes
    )
    output_csv_multi = base_results_dir / "multi_layer_probe_metrics.csv"

    df_metrics_multi = train_linear_probe(
        linear_probe=multi_layer_probe,
        train_loader=train_loader,
        val_loader=val_loader,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        output_csv=str(output_csv_multi),
        beta_value=beta,
        model_name=model_name + "_TwoLayer",
    )

    print(f"[INFO] Multi-layer probe metrics saved to: {output_csv_multi}")
    plot_path_multi = base_results_dir / "multi_layer_val_accuracy_curve.png"
    plot_validation_curve(
        df_metrics_multi,
        save_path=plot_path_multi,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        beta=beta,
    )
    print("[INFO] Multi-layer probe training completed.")

    ########################################################################
    #                     Combined Accuracy Plot                          #
    ########################################################################
    # Add probe labels
    df_metrics["Probe"] = "Linear"
    df_metrics_multi["Probe"] = "TwoLayer"
    df_combined = pd.concat([df_metrics, df_metrics_multi], ignore_index=True)

    # Save combined plot
    combined_plot_path = base_results_dir / "combined_val_accuracy_curve.png"
    plot_combined_validation_curve(
        df_combined,
        save_path=combined_plot_path,
        source_timestep=source_timestep,
        target_timestep=target_timestep,
        beta=beta,
    )
    print(f"[INFO] Combined accuracy plot saved to: {combined_plot_path}")

    ########################################################################
    #                Save PCA projections and components                 #
    ########################################################################
    np.save(base_results_dir / "pca_latents.npy", pca_latents)
    np.save(base_results_dir / "pca_labels_mapped.npy", all_labels_mapped)
    np.save(base_results_dir / "pca_labels_original.npy", all_labels)
    np.save(base_results_dir / "pca_components.npy", pca.components_)
    np.save(base_results_dir / "explained_variance.npy", pca.explained_variance_ratio_)

    print("\n[INFO] PCA evaluation completed.")
    print(f"[INFO] Metrics CSV: {output_csv}")
    print(f"[INFO] Validation curve plot: {plot_path}")

    ########################################################################
    #                    Plot PCA 2D projection                            #
    ########################################################################
    print("\n--- Plotting PCA 2D Projection ---")
    pca_2d_path = base_results_dir / "pca_2d_projection.png"
    plot_pca_2d_projection(
        pca_latents=pca_latents,
        labels=all_labels_mapped,
        save_path=pca_2d_path,
        title="PCA 2D Projection (First 2 components, mapped labels)",
    )

    print("\n--- Plotting PCA 2D Projection (Original Labels) ---")
    pca_2d_path_original = base_results_dir / "pca_2d_projection_original_labels.png"
    plot_pca_2d_projection(
        pca_latents=pca_latents,
        labels=all_labels,
        save_path=pca_2d_path_original,
        title="PCA 2D Projection (First 2 components, original labels)",
    )

    ########################################################################
    #                    k-Means Evaluation & Plotting                     #
    ########################################################################
    print("\n--- Evaluating k-Means clustering ---")
    clustering_results = []
    k_range = [3, 5, 10, 25, 50, 80]

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_latents)

        sil_score = silhouette_score(pca_latents, cluster_labels)
        ch_score = calinski_harabasz_score(pca_latents, cluster_labels)
        db_score = davies_bouldin_score(pca_latents, cluster_labels)

        clustering_results.append(
            {
                "k": k,
                "Silhouette": sil_score,
                "Calinski-Harabasz": ch_score,
                "Davies-Bouldin": db_score,
            }
        )

        print(
            f"[k={k}] Silhouette: {sil_score:.3f}, CH: {ch_score:.1f}, DB: {db_score:.3f}"
        )

    # Save metrics as CSV
    cluster_metrics_df = pd.DataFrame(clustering_results)
    cluster_metrics_path = base_results_dir / "kmeans_clustering_metrics.csv"
    cluster_metrics_df.to_csv(cluster_metrics_path, index=False)
    print(f"[INFO] k-Means clustering metrics saved to: {cluster_metrics_path}")

    # Plot 2D PCA cluster grid
    kmeans_grid_path = base_results_dir / "kmeans_pca_grid.png"
    plot_kmeans_grid(
        pca_latents=pca_latents,
        k_values=k_range,
        max_data_samples=max_umap_samples,
        save_to_path=kmeans_grid_path,
    )
    print(f"[INFO] k-Means PCA grid plot saved to: {kmeans_grid_path}")

    ########################################################################
    #                    Plot Latent Histograms                            #
    ########################################################################
    print("\n--- Plotting Latent Histograms ---")
    plot_latent_histograms(pca_latents=pca_latents, save_dir=base_results_dir)

    ########################################################################
    #                  Plot UMAP projections                              #
    ########################################################################
    print("\n--- Plotting UMAP Projection (original labels) ---")
    plot_umap_pca(
        pca_latents=pca_latents,
        labels=all_labels,
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        max_data_samples=max_umap_samples,
        save_to_path=base_results_dir / "umap_plot_original_labels.png",
        title="UMAP of PCA-projected Latents (Original Labels)",
    )

    print("\n--- Plotting UMAP Projection (mapped labels) ---")
    plot_umap_pca(
        pca_latents=pca_latents,
        labels=all_labels_mapped,
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        max_data_samples=max_umap_samples,
        save_to_path=base_results_dir / "umap_plot_mapped_labels.png",
        title="UMAP of PCA-projected Latents (Mapped Labels)",
    )

    print("\n[INFO] All plots and artifacts saved.")
    return (
        df_metrics,
        df_metrics_multi,
        pca_latents,
        all_labels_mapped,
        pca.components_,
        pca.explained_variance_ratio_,
    )


###########################################
#        Main Execution Block             #
###########################################

if __name__ == "__main__":
    #####################################
    # Shared Parameters
    #####################################
    dataset_name = "imagenet256-dataset-T000006"
    group = "validation"
    num_components = 20
    max_data_samples = 100000
    batch_size = 64
    data_path = "./dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5"

    n_neighbors = 20
    min_dist = 0.1
    max_umap_samples = 25000
    random_state = 42  # state of life
    epochs = 500
    patience = 10
    lr = 1e-4

    #####################################
    # Device + Seed Setup
    #####################################
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()

    #####################################
    # Define ALL models with target = 1.0
    #####################################

    # All models under Reconstruction Objective
    # -------------------------------------------------------
    model_configs_v0 = [
        # beta: 0.1
        {
            "name": "Beta05x05x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta00x00x_01b",
            "beta": 0.1,
            "source_ts": 0.00,
            "target_ts": 0.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt",
        },  # Open
        {
            "name": "Beta00x00x_01b",
            "beta": 0.1,
            "source_ts": 0.20,
            "target_ts": 0.20,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt",
        },  # Open
        # beta: 1.0
        {
            "name": "Beta05x05x_1b",
            "beta": 1.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt",
        },  # Open
        # beta: 5.0
        {
            "name": "Beta05x05x_5b",
            "beta": 5.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt",
        },  # Open
    ]

    # All models under Denoising Objective
    # -------------------------------------------------------
    # All models with b:0.1
    model_configs_v1 = [
        # beta: 0.1
        {
            "name": "Beta05x05x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta05x10x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-02/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta04x10x_01b",
            "beta": 0.1,
            "source_ts": 0.40,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.40x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta03x10x_01b",
            "beta": 0.1,
            "source_ts": 0.30,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.30x-1.00x-0.1b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_01b",
            "beta": 0.1,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta00x10x_01b",
            "beta": 0.1,
            "source_ts": 0.00,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-1.00x-0.1b/2025-06-18/29852/V0-eV2/2025-06-24/29852/checkpoints/last.ckpt",
        },
    ]

    # All models with b:1.0
    model_configs_v2 = [
        # beta: 1.0
        {
            "name": "Beta05x05x_1b",
            "beta": 1.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta05x10x_1b",
            "beta": 1.0,
            "source_ts": 0.50,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-1.0b/2025-06-21/manual/V2/2025-06-21/29807/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_1b",
            "beta": 1.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt",
        },
    ]

    # All models with b:5.0
    model_configs_v3 = [
        # beta: 5.0
        {
            "name": "Beta05x05x_5b",
            "beta": 5.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta05x10x_5b",
            "beta": 5.0,
            "source_ts": 0.50,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-5.0b/2025-06-21/manual/V2/2025-06-21/101101/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_5b",
            "beta": 5.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt",
        },
    ]

    # All models with fixed beta:0.1
    model_configs_v4 = [
        # Self-reconstruction tasks
        {
            "name": "Beta05x05x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta02x02x_01b",
            "beta": 0.1,
            "source_ts": 0.20,
            "target_ts": 0.20,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-0.20x-0.1b/2025-06-18/29842/V2/2025-06-18/29842/checkpoints/last.ckpt",
        },
        {
            "name": "Beta00x00x_01b",
            "beta": 0.1,
            "source_ts": 0.00,
            "target_ts": 0.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.00x-0.00x-0.1b/2025-06-11/29845/checkpoints/last.ckpt",
        },
    ]

    model_configs_v5 = [
        # comparison of all baseline models with source: 0.5 -> target 0.5
        {
            "name": "Beta05x05x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-0.1b/2025-06-18/29847/V2/2025-06-18/29847/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta05x05x_1b",
            "beta": 1.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.50x-0.50x-1.0b/2025-06-17/29850/checkpoints/last.ckpt",
        },  # Open (Baseline)
        {
            "name": "Beta05x05x_5b",
            "beta": 5.0,
            "source_ts": 0.50,
            "target_ts": 0.50,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-0.50x-5.0b/2025-06-21/manual/V2/2025-06-21/29852/checkpoints/last.ckpt",
        },  # Open (Baseline)
    ]

    # All models with source: 2.0 --> target: 1.0
    model_configs_v6 = [
        # mixed beta = {1e-4, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0}
        {
            "name": "Beta02x10x_1e4b",
            "beta": 1e-4,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_01b",
            "beta": 0.1,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-0.1b/2025-06-21/manual/V0/2025-07-06/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_05b",
            "beta": 0.5,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_1b",
            "beta": 1.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-1.0b/2025-06-17/29812/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_2b",
            "beta": 2.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-2.0b/V2/2025-07-06/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_3b",
            "beta": 3.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.00x-3.0b/2025-06-21/manual/V0/2025-06-30/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta02x10x_5b",
            "beta": 5.0,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.0x-5.0b/2025-06-21/manual/V0/2025-07-02/101646/checkpoints/last.ckpt",
        },
    ]

    #####################################
    # Put all groups in a list
    #####################################
    all_config_groups = [
        {"group_name": "Beta0.1_models", "configs": model_configs_v1},
        {"group_name": "Beta1.0_models", "configs": model_configs_v2},
        {"group_name": "Beta5.0_models", "configs": model_configs_v3},
        {
            "group_name": "Beta0.1_ReconstructionTask_models",
            "configs": model_configs_v4,
        },
        {"group_name": "Beta0.1_Baseline_models", "configs": model_configs_v5},
        {"group_name": "Source0.2_DenoisingTask_models", "configs": model_configs_v6},
    ]

    #####################################
    # Loop through each group automatically
    #####################################

    for config_group in all_config_groups:
        group_name = config_group["group_name"]
        configs = config_group["configs"]
        all_metrics = []
        all_multi_metrics = []
        model_results = []

        print(f"\n=== Running group: {group_name} ===")

        for config in configs:
            beta = config["beta"]
            checkpoint = config["ckpt"]
            source_ts = config["source_ts"]
            target_ts = config["target_ts"]
            model_tag = config["name"]
            project_name = f"PCA_Quantitative_{group_name}"
            model_name = f"{model_tag}_{dataset_name}"

            print(
                f"\n[INFO] Running model: {model_tag} (β={beta}, source={source_ts:.2f}, target={target_ts:.2f})"
            )

            df_metrics, df_multi_metrics, *_ = run_pca_eval(
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
                max_data_samples=max_data_samples,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lr=lr,
                results_root="results",
            )

            # Save metrics
            df_metrics["Model"] = model_tag
            df_metrics["Beta"] = beta
            df_metrics["Source_TS"] = source_ts
            df_metrics["Target_TS"] = target_ts
            all_metrics.append(df_metrics)

            df_multi_metrics["Model"] = model_tag
            df_multi_metrics["Beta"] = beta
            df_multi_metrics["Source_TS"] = source_ts
            df_multi_metrics["Target_TS"] = target_ts
            all_multi_metrics.append(df_multi_metrics)

            print(
                f"[INFO] Metrics for {model_tag} saved to: results/{project_name}/{model_name}/linear_probe_metrics.csv"
            )

            # ----------------------------------------------------------
            # Load PCA latents and labels for UMAP grid
            # ----------------------------------------------------------
            results_dir = Path("results") / project_name / model_name
            timestamp_dirs = list(results_dir.glob("*"))
            if not timestamp_dirs:
                print(
                    f"[WARN] No results found for model: {model_tag}, skipping for UMAP grid."
                )
                continue

            latest_dir = max(timestamp_dirs, key=os.path.getmtime)

            latents_path = latest_dir / "pca_latents.npy"
            labels_path = (
                latest_dir / "pca_labels_original.npy"
            )  # or "pca_labels_mapped.npy" if desired

            if not latents_path.exists() or not labels_path.exists():
                print(
                    f"[WARN] Missing PCA files for model: {model_tag}, skipping UMAP grid."
                )
                continue

            latents = np.load(latents_path)
            labels = np.load(labels_path)

            model_results.append(
                {"latents": latents, "labels": labels, "name": model_tag}
            )

        print(f"\n[INFO] Completed group: {group_name} with {len(all_metrics)} models.")

        # ----------------------------------------------------------
        # Save combined metrics plot
        # ----------------------------------------------------------
        if all_metrics:
            combined_df = pd.concat(all_metrics, ignore_index=True)

            plt.figure(figsize=(14, 8))
            sns.set(style="whitegrid", font="serif")
            palette = sns.color_palette(
                "crest", n_colors=len(combined_df["Model"].unique())
            )

            for model_tag, color in zip(combined_df["Model"].unique(), palette):
                model_df = combined_df[combined_df["Model"] == model_tag]
                plt.plot(
                    model_df["Epoch"],
                    model_df["Val_Accuracy"],
                    marker="o",
                    linewidth=2,
                    label=model_tag,
                    color=color,
                )

            plt.title(
                f"Validation Accuracy Comparison — {group_name}", fontsize=18, pad=20
            )
            plt.xlabel("Epoch", fontsize=15)
            plt.ylabel("Validation Accuracy", fontsize=15)
            plt.ylim(0, 0.5)
            plt.legend(title="Model", fontsize=10, loc="upper right")
            plt.grid(True, linestyle=":", alpha=0.6)

            combined_plot_path = f"combined_validation_curve_{group_name}.png"
            plt.savefig(combined_plot_path, bbox_inches="tight", dpi=300)
            print(f"\n[INFO] Combined validation plot saved to: {combined_plot_path}")
            plt.show()

        if all_multi_metrics:
            combined_multi_df = pd.concat(all_multi_metrics, ignore_index=True)

            plt.figure(figsize=(14, 8))
            sns.set(style="whitegrid", font="serif")
            palette = sns.color_palette(
                "crest", n_colors=len(combined_multi_df["Model"].unique())
            )

            for model_tag, color in zip(combined_multi_df["Model"].unique(), palette):
                model_df = combined_multi_df[combined_multi_df["Model"] == model_tag]
                plt.plot(
                    model_df["Epoch"],
                    model_df["Val_Accuracy"],
                    marker="o",
                    linewidth=2,
                    label=model_tag,
                    color=color,
                )

            plt.title(
                f"Validation Accuracy Comparison (Multi-Layer) — {group_name}",
                fontsize=18,
                pad=20,
            )
            plt.xlabel("Epoch", fontsize=15)
            plt.ylabel("Validation Accuracy", fontsize=15)
            plt.ylim(0, 0.5)
            plt.legend(title="Model", fontsize=10, loc="upper right")
            plt.grid(True, linestyle=":", alpha=0.6)

            combined_multi_plot_path = (
                f"combined_validation_curve_multi_{group_name}.png"
            )
            plt.savefig(combined_multi_plot_path, bbox_inches="tight", dpi=300)
            print(
                f"\n[INFO] Combined multi-layer validation plot saved to: {combined_multi_plot_path}"
            )
            plt.show()

        # Generate UMAP grid plot
        if model_results:
            umap_grid_save_path = f"umap_grid_{group_name}.png"
            plot_umap_grid(
                model_results,
                group_name=group_name,
                save_path=umap_grid_save_path,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=num_components,
                max_data_samples=max_umap_samples,
                random_state=random_state,
            )

    # CUDA_VISIBLE_DEVICES=2 python ...
