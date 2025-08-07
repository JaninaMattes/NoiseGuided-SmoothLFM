import os
import sys
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from scipy.stats import entropy
from lightning import seed_everything

from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif

import scipy.stats as stats
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




from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split






# Helper utilities
from jutils import freeze

# Project root path setup
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")
)
sys.path.append(project_root)

# Project-specific modules
from ldm.trainer_bvae_ti2 import TrainerModuleLatentBetaVae
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule

# Torch precision
torch.set_float32_matmul_precision("high")


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
    max_samples=50000,
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
    dataloader = (
        data.val_dataloader() if group == "validation" else data.test_dataloader()
    )

    # Results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

            source_latents = batch[f"latents_{source_timestep:.2f}"].to(
                device, non_blocking=True
            )
            encoded = beta_vae_module.model.encode(source_latents)
            latents = encoded["latent_dist"].mode()
            all_latents.append(latents.detach().cpu().numpy())

            if "label" in batch:
                all_labels.append(batch["label"].detach().cpu().numpy())
            curr_samples += latents.shape[0]

    all_latents = np.vstack(all_latents)

    if all_labels is not None:
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_labels = None
        print("[WARNING] No labels found in dataset. Proceeding without labels.")

    print(f"[INFO] Collected latents shape: {all_latents.shape}")
    print(
        f"[INFO] Collected labels shape: {all_labels.shape if all_labels is not None else 'N/A'}"
    )

    return all_latents, all_labels


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
def train_sklearn_probe(model, X_train, y_train, X_val, y_val, max_iter=500):
    """
    Trains a scikit-learn classifier and returns validation accuracy + metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return {"ValAccuracies": acc, "Model": model.__class__.__name__}


def train_custom_probe(
    linear_probe,
    train_loader,
    val_loader,
    source_timestep,
    target_timestep,
    label_key="label",
    latent_key="pca",
    device=None,
    epochs=500,
    patience=10,
    lr=1e-4,
    output_csv="linear_probe_metrics.csv",
    beta_value=1e-4,  # default low beta value for β-VAE
    model_name="",
    output_dim=90,
):
    """Train a custom linear probe on PCA latents."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def evaluate_sklearn_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    return {
        "ValAccuracies": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, average="macro"),
        "Recall": recall_score(y_val, y_pred, average="macro"),
        "F1Score": f1_score(y_val, y_pred, average="macro"),
        "ConfusionMatrix": confusion_matrix(y_val, y_pred).tolist(),
        "ClassificationReport": classification_report(y_val, y_pred, output_dict=True),
    }


###########################################################
#                   PCA Variance Plots                    #
###########################################################
def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Cambria", "DejaVu Serif"],
            "axes.facecolor": "#f5f5f5",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "grid.linestyle": ":",
            "grid.color": "grey",
            "figure.figsize": (10, 6),
            "grid.alpha": 0.5,
            "lines.linewidth": 2.0,
        }
    )
    sns.set_palette("Set2")
    print("[INFO] Matplotlib style set for PCA plots.")


def plot_scree(explained_variance, threshold=1.0, save_path=None):
    """
    Plot a scree plot showing eigenvalues by principal component.

    Args:
        explained_variance (list or np.array): Eigenvalues from PCA.
        threshold (float): Optional horizontal threshold line.
        save_path (str): Optional path to save the figure.
    """
    pcs = np.arange(1, len(explained_variance) + 1)
    df = pd.DataFrame({"PC": pcs, "Eigenvalue": explained_variance})

    plt.figure(figsize=(10, 4))
    sns.lineplot(
        data=df, x="PC", y="Eigenvalue", marker="o", color="mediumseagreen", linewidth=2
    )
    plt.axhline(
        y=threshold,
        linestyle="--",
        color="gray",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    # Control x-ticks (every 5th component or fewer)
    max_ticks = 10
    tick_step = max(1, len(pcs) // max_ticks)
    plt.xticks(pcs[::tick_step])

    plt.title("Scree Plot: Eigenvalues by Principal Component", fontsize=14)
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Eigenvalue", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()




def plot_variance_explained(explained_variance_ratio, save_path=None):
    """
    Plot explained variance and cumulative variance for PCA components.
    """
    pcs = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)

    df = pd.DataFrame(
        {"PC": pcs, "Explained": explained_variance_ratio, "Cumulative": cumulative}
    )

    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=df, x="PC", y="Explained", color="skyblue", alpha=0.6)
    sns.lineplot(
        data=df, x="PC", y="Cumulative", marker="o", color="blue", label="Cumulative"
    )

    # Optional: remove dense per-bar annotations
    # for i, value in enumerate(explained_variance_ratio):
    #     if i % 5 == 0:  # Only annotate every 5 bars
    #         ax.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

    # Set fewer x-ticks to avoid clutter
    max_ticks = 12
    tick_step = max(1, len(pcs) // max_ticks)
    ax.set_xticks(pcs[::tick_step])
    ax.set_xticklabels(pcs[::tick_step])

    plt.axhline(0.9, linestyle="--", color="gray", linewidth=2, label="90% Threshold")
    plt.title("Explained Variance by PCA Components", fontsize=14)
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Variance Ratio", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_pca_scatter(pca_latents, labels, num_components=2, save_path=None):
    """
    Plot a 2D scatter plot of PCA latents with labels.
    """
    set_style()

    if num_components < 2:
        raise ValueError("Need at least 2 components to plot scatter.")

    df = pd.DataFrame(pca_latents[:, :2], columns=["PC1", "PC2"])
    df["Label"] = labels

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df, x="PC1", y="PC2", hue="Label", palette="tab10", alpha=0.7, s=40
    )
    plt.title("PCA Scatter Plot (2D)")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_decision_boundary(X, y, clf, title="Decision Boundary", save_path=None):

    set_style()

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="tab10", s=30, edgecolor="k")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_svm_decision_boundary_2d(pca_latents, labels, model, save_path=None):
    """
    Plot SVM decision boundary in 2D PCA space.
    """
    set_style()

    # ---- FIX: ensure labels are 1D ----
    labels = np.array(labels).ravel()

    x_min, x_max = pca_latents[:, 0].min() - 1, pca_latents[:, 0].max() + 1
    y_min, y_max = pca_latents[:, 1].min() - 1, pca_latents[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")

    # Ensure scatterplot input is clean
    sns.scatterplot(
        x=pca_latents[:, 0],
        y=pca_latents[:, 1],
        hue=labels,
        palette="tab10",
        s=40,
        edgecolor="k",
    )

    plt.title("SVM Decision Boundary in PCA Space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def class_entropy_by_pc(pca_latents, labels, num_bins=20):
    """
    Compute entropy of class distribution for each PCA component.
    """
    # Ensure labels is a flattened 1D array of ints
    labels = np.asarray(labels).astype(int).flatten()
    assert np.all(labels >= 0), "Labels contain negative values!"

    results = []
    for i in range(pca_latents.shape[1]):
        pc_values = pca_latents[:, i].astype(np.float64)
        binned = pd.cut(pc_values, bins=num_bins, labels=False).astype(int)

        pc_entropy = 0
        for b in np.unique(binned):
            idx = np.where(binned == b)[0]
            if len(idx) == 0:
                continue  # skip empty bins

            label_counts = np.bincount(labels[idx], minlength=np.max(labels) + 1)
            probs = label_counts / (label_counts.sum() + 1e-8)
            pc_entropy += entropy(probs) * (len(idx) / len(binned))
        results.append(pc_entropy)

    return results


def compute_latent_distribution_stats(pca_latents, prefix="PC"):
    """
    Compute mean, std, skewness, and kurtosis for each PCA component.
    """
    stats_dict = {
        "Mean": np.mean(pca_latents, axis=0).tolist(),
        "StdDev": np.std(pca_latents, axis=0).tolist(),
        "Skewness": skew(pca_latents, axis=0).tolist(),
        "Kurtosis": kurtosis(pca_latents, axis=0).tolist(),
    }

    return stats_dict


def compute_pc_mutual_information(pca_latents, labels, random_state=2025):
    """
    Computes mutual information between each PC and class labels.
    """
    mi_scores = mutual_info_classif(
        pca_latents, labels, discrete_features=False, random_state=random_state
    )
    return mi_scores.tolist()


def run_pca_over_beta(
    source_timestep=0.20,
    target_timestep=1.00,
    beta=1.0,
    dataset_name="imagenet256-dataset",
    group="validation",
    checkpoint=None,
    data_path=None,
    project_path=None,
    model_name=None,
    num_components=20,
    max_data_samples=50000,
    max_umap_samples=20000,
    pca_latent_numbers=[2, 3, 5],
    batch_size=32,
    epochs=500,
    patience=10,
    lr=1e-4,
    device=None,
    results_root=None,
):

    # Ensure path types
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

    # Load β-VAE model
    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    beta_vae_module.eval().to(device)
    freeze(beta_vae_module.model)

    # Collect latent representations
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
        max_samples=max_data_samples,
    )

    # Prepare label mapping before use
    unique_labels = np.unique(all_labels)
    label_map = {int(lbl): int(idx) for idx, lbl in enumerate(sorted(unique_labels))}
    inverse_label_map = {v: k for k, v in label_map.items()}
    all_labels_mapped = np.vectorize(label_map.get)(all_labels)
    num_classes = len(unique_labels)

    # Save label mappings
    base_results_dir = project_path / f"pca_evaluation_{model_name}"
    base_results_dir.mkdir(parents=True, exist_ok=True)
    with (base_results_dir / "label_map.json").open("w") as f:
        json.dump(label_map, f, indent=2)
    with (base_results_dir / "inverse_label_map.json").open("w") as f:
        json.dump(inverse_label_map, f, indent=2)

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Save PCA metrics
    pca_metrics_df = pd.DataFrame(
        {
            "PC": np.arange(1, len(explained_variance_ratio) + 1),
            "ExplainedVariance": explained_variance,
            "ExplainedVarianceRatio": explained_variance_ratio,
            "CumulativeVarianceRatio": cumulative_variance_ratio,
        }
    )
    pca_metrics_df.to_csv(
        base_results_dir / f"pca_metrics_{model_name}.csv", index=False
    )

    # Generate plots
    print("[INFO] Plotting PCA scree and variance plots...")
    plot_scree(
        explained_variance, save_path=base_results_dir / f"pca_scree_{model_name}.png"
    )
    plot_variance_explained(
        explained_variance_ratio,
        save_path=base_results_dir / f"pca_variance_{model_name}.png",
    )
    if num_components >= 2:
        plot_pca_scatter(
            pca_latents,
            all_labels,
            num_components=2,
            save_path=base_results_dir / f"pca_scatter_2d_{model_name}.png",
        )

    # --- Extended PCA Analysis ---
    print("[INFO] Computing extended PCA statistics...")
    try:
        latent_stats = compute_latent_distribution_stats(pca_latents)
    except Exception as e:
        latent_stats = {
            "Skewness": [0] * num_components,
            "Kurtosis": [0] * num_components,
            "Mean": [0] * num_components,
            "StdDev": [0] * num_components,
        }
        print(f"[ERROR] Failed to compute latent stats: {e}")

    try:
        pc_mi_scores = compute_pc_mutual_information(pca_latents, all_labels_mapped)
    except Exception as e:
        print(f"[ERROR] Failed to compute mutual information: {e}")
        pc_mi_scores = [0] * num_components
    except ValueError as ve:
        print(f"[ERROR] ValueError in mutual information computation: {ve}")
        pc_mi_scores = [0] * num_components

    try:
        pc_class_entropy = class_entropy_by_pc(pca_latents, all_labels_mapped)
    except Exception as e:
        print(f"[ERROR] Failed to compute class entropy: {e}")
        pc_class_entropy = [0] * num_components

    # --- Save extended PCA analysis
    pd.DataFrame(
        {
            "PC": np.arange(1, num_components + 1),
            "MutualInformation": pc_mi_scores,
            "ClassEntropy": pc_class_entropy,
            "Skewness": latent_stats["Skewness"],
            "Kurtosis": latent_stats["Kurtosis"],
        }
    ).to_csv(base_results_dir / "pca_pc_analysis.csv", index=False)

    structured_results = {
        "Model": model_name,
        "Beta": beta,
        "Source_TS": source_timestep,
        "Target_TS": target_timestep,
        "NumSamples": int(len(all_latents)),
        "NumClasses": int(num_classes),
        "NumComponents": int(num_components),
        "ExplainedVariance": explained_variance.tolist(),
        "ExplainedVarianceRatio": explained_variance_ratio.tolist(),
        "CumulativeVarianceRatio": cumulative_variance_ratio.tolist(),
        "PCA_Latent_Stats": {
            "Mean": np.mean(pca_latents, axis=0).tolist(),
            "StdDev": np.std(pca_latents, axis=0).tolist(),
            "Skewness": latent_stats["Skewness"],
            "Kurtosis": latent_stats["Kurtosis"],
        },
        "PC_MutualInformation": pc_mi_scores,
        "PC_Entropy": pc_class_entropy,
        "Results": [],
    }

    # -------- Linear probes -----------
    for pca_num in pca_latent_numbers:
        if pca_num > num_components:
            continue

        print(f"\n--- Evaluating PCA with {pca_num} components ---")
        result_dir = base_results_dir / f"pca_{pca_num}"
        result_dir.mkdir(parents=True, exist_ok=True)

        pca_latents_subset = pca_latents[:, :pca_num]
        X_train, X_val, y_train, y_val = train_test_split(
            pca_latents_subset,
            all_labels_mapped,
            test_size=0.2,
            random_state=2025,
            stratify=all_labels_mapped,
        )
        train_loader = DataLoader(
            PCADataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            PCADataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )

        # SVM
        print("Training SVM Probe...")
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", max_iter=1000)
        svm.fit(X_train, y_train)
        svm_metrics = evaluate_sklearn_model(svm, X_val, y_val)
        structured_results["Results"].append(
            {
                "ProbeType": "SVM",
                "PCA": pca_num,
                **svm_metrics,
                "NumSupportVectors": int(len(svm.support_)),
                "SupportVectorsPerClass": svm.n_support_.tolist(),
            }
        )
        x_val_2d = X_val[:, :2] if pca_num >= 2 else X_val

        try:
            if pca_num >= 2:
                plot_svm_decision_boundary_2d(
                    x_val_2d,
                    y_val,
                    svm,
                    save_path=result_dir / "svm_decision_boundary.png",
                )
            else:
                print("[WARNING] Not enough components for 2D SVM plot. Skipping...")
        except Exception as e:
            print(f"[ERROR] Failed to plot SVM decision boundary: {e}")

        # Linear SVM
        print("Training Linear SVM Probe...")
        linear_svm = LinearSVC(C=1.0, max_iter=1000, random_state=2025)
        linear_svm.fit(X_train, y_train)
        lsvm_metrics = evaluate_sklearn_model(linear_svm, X_val, y_val)
        coef_matrix = np.array(linear_svm.coef_)
        importance = np.mean(np.abs(coef_matrix), axis=0)
        correlation = stats.pearsonr(
            explained_variance_ratio[: len(importance)], importance
        )

        structured_results["Results"].append(
            {
                "ProbeType": "LinearSVM",
                "PCA": pca_num,
                **lsvm_metrics,
                "Coefficients": coef_matrix.tolist(),
                "Intercepts": linear_svm.intercept_.tolist(),
                "ImportanceMeanAbs": importance.tolist(),
                "VarianceImportanceCorrelation": {
                    "PearsonR": correlation[0],
                    "PValue": correlation[1],
                },
            }
        )
        print(f"[INFO] Linear SVM Coefficients: {coef_matrix}")

        try:
            if pca_num >= 2:
                plot_decision_boundary(
                    x_val_2d,
                    y_val,
                    linear_svm,
                    title="Linear SVM Decision Boundary",
                    save_path=result_dir / "linear_svm_decision_boundary.png",
                )
            else:
                print(
                    "[WARNING] Not enough components for 2D Linear SVM plot. Skipping..."
                )
        except Exception as e:
            print(f"[ERROR] Failed to plot Linear SVM decision boundary: {e}")

        # MLP
        print("Training MLP Probe...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            early_stopping=True,
            random_state=2025,
        )
        mlp.fit(X_train, y_train)
        mlp_metrics = evaluate_sklearn_model(mlp, X_val, y_val)
        structured_results["Results"].append(
            {
                "ProbeType": "MLP",
                "PCA": pca_num,
                **mlp_metrics,
                "LossCurve": mlp.loss_curve_,
                "NumIterations": int(mlp.n_iter_),
            }
        )

        # Linear Probe
        print("Training Linear Probe...")
        linear_probe = LinearProbe(hidden_size=pca_num, num_classes=num_classes)
        df_linear = train_custom_probe(
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
            model_name=model_name,
        )
        structured_results["Results"].append(
            {
                "ProbeType": "Linear",
                "PCA": pca_num,
                "ValAccuracies": df_linear["Val_Accuracy"].tolist(),
                "BestValAccuracies": float(df_linear["Val_Accuracy"].max()),
                "MeanValAccuracies": float(df_linear["Val_Accuracy"].mean()),
                "NumEpochs": len(df_linear),
            }
        )

        # Two-Layer Probe
        print("Training Two-Layer Probe...")
        two_layer_probe = TwoLayerProbe(
            input_dim=pca_num, hidden_dim=128, num_classes=num_classes
        )
        df_two = train_custom_probe(
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
            model_name=model_name + "_TwoLayer",
        )
        structured_results["Results"].append(
            {
                "ProbeType": "Two-Layer",
                "PCA": pca_num,
                "ValAccuracies": df_two["Val_Accuracy"].tolist(),
                "BestValAccuracies": float(df_two["Val_Accuracy"].max()),
                "MeanValAccuracies": float(df_two["Val_Accuracy"].mean()),
                "NumEpochs": len(df_two),
            }
        )

        # Save PCA data
        np.save(result_dir / "pca_latents.npy", pca_latents_subset)
        np.save(result_dir / "pca_labels_mapped.npy", y_val)
        np.save(result_dir / "pca_labels_original.npy", all_labels)

    # Save structured results
    with (base_results_dir / "structured_results.json").open("w") as f:
        json.dump(structured_results, f, indent=2)

    return structured_results


######################################################################
#                   Full PCA Evaluation Pipeline                      #
######################################################################


def run_pca_for_kmeans_and_plot_scatter_grid(
    source_timestep=0.20,
    target_timestep=1.00,
    beta=1.0,
    dataset_name="imagenet256-dataset",
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
    results_root="results",
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(2025)
    torch.cuda.empty_cache()
    gc.collect()

    beta_vae_module = TrainerModuleLatentBetaVae.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
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
        max_samples=max_data_samples,
    )

    pca = PCA(n_components=num_components)
    pca_latents = pca.fit_transform(all_latents)

    scatter_data = []
    all_kmeans_metrics = []

    for pca_num in pca_latent_numbers:
        if pca_num < 2 or pca_num > num_components:
            continue

        pca_subset = pca_latents[:, :pca_num]

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=2025)
            clusters = kmeans.fit_predict(pca_subset)

            # --- Metric Calculation ---
            metrics = {
                "Silhouette": silhouette_score(pca_subset, clusters),
                "CalinskiHarabasz": calinski_harabasz_score(pca_subset, clusters),
                "DaviesBouldin": davies_bouldin_score(pca_subset, clusters),
            }

            if all_labels is not None:
                true_labels_subset = all_labels[: len(clusters)]
                metrics.update(
                    {
                        "NMI": normalized_mutual_info_score(
                            true_labels_subset, clusters
                        ),
                        "ARI": adjusted_rand_score(true_labels_subset, clusters),
                    }
                )

            all_kmeans_metrics.append({"PCA": pca_num, "K": k, "Metrics": metrics})

            # --- Scatter Plot Data ---
            for i in range(len(pca_subset)):
                scatter_data.append(
                    {
                        "x": pca_subset[i, 0],
                        "y": pca_subset[i, 1],
                        "PCA": pca_num,
                        "K": k,
                        "Cluster": int(clusters[i]),  # ensure JSON serializable
                    }
                )

    # Save scatter plot
    df_scatter = pd.DataFrame(scatter_data)
    plot_pca_kmeans_scatter_grid(
        df_scatter, project_path=project_path or "test_outputs"
    )

    # Save metrics JSON
    output_dir = Path(project_path or "test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "kmeans_metrics_pca.json"

    with open(metrics_path, "w") as f:
        json.dump(all_kmeans_metrics, f, indent=2)
    print(f"[INFO] Saved K-Means metrics to: {metrics_path}")


def plot_pca_kmeans_scatter_grid(df, project_path="test_outputs"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
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
        }
    )

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
            data=data,
            x="x",
            y="y",
            hue="Cluster",
            palette="tab10",
            s=20,
            linewidth=0,
            alpha=0.8,
            legend=False,
            ax=ax,
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


if __name__ == "__main__":
    #####################################
    # Shared Parameters
    #####################################
    dataset_name = "imagenet256-dataset-T000006"
    group = "validation"
    num_components = 50
    max_data_samples = 100000
    batch_size = 64
    data_path = "./dataset/processed/trainset-256/imagenet256-dataset-T000006.hdf5"
    results_path = "./results/PCA_BetaVAE_Eval"

    n_neighbors = 50
    min_dist = 0.1
    max_umap_samples = 25000
    random_state = 2025  # state of life
    epochs = 500
    patience = 3
    lr = 1e-4

    pca_latent_numbers = [2, 3, 5, 9, 15, 20, 30, 50]
    k_means_n_values = [2, 3, 5, 9, 15, 20, 30, 50]

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
    experiment_root = Path(results_path) / timestamp  #'./results/PCA_Quantitative_Eval'
    experiment_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {experiment_root}")

    # --------------------------------------
    # Model Configurations
    # --------------------------------------

    # Define the original configurations as nested dictionaries (grouped by v1, v2, v3, etc.)
    all_model_configs = [
        # From model_configs_v2 (beta=0.1)
        {
            "name": "Beta02x10x_1e4b",
            "beta": 1e-4,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v0/0.20x-1.00x-0.0001b/2025-06-21/manual/V0/2025-06-27/101646/checkpoints/last.ckpt",
        },
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
        {
            "name": "Beta05x10x_01b",
            "beta": 0.1,
            "source_ts": 0.50,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-08-04/manual/V2/2025-08-04/100001/checkpoints/last.ckpt",
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
        {
            "name": "Beta02x10x_05b",
            "beta": 0.5,
            "source_ts": 0.20,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.20x-1.0x-0.5b/2025-06-30/manual/V2/2025-07-03/101646/checkpoints/last.ckpt",
        },
        {
            "name": "Beta05x10x_05b",
            "beta": 0.5,
            "source_ts": 0.50,
            "target_ts": 1.00,
            "ckpt": "./logs_dir/imnet256/beta-vae-skipViT-b-2/imagenet256_hdf5_v2/0.50x-1.00x-0.1b/2025-06-30-1435/manual/V2/2025-07-31/101646/checkpoints/last.ckpt",
        },
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

    all_config_groups = [
        {"group_name": "All_Beta_Models", "configs": all_model_configs},
    ]

    for config_group in all_config_groups:
        group_name = config_group["group_name"]
        configs = config_group["configs"]
        model_results = []

        print(f"\n=== Running group: {group_name} ===")

        for config in configs:
            beta = config["beta"]
            checkpoint = config["ckpt"]
            source_ts = config["source_ts"]
            target_ts = config["target_ts"]
            model_tag = config["name"]
            model_name = f"{model_tag}_{dataset_name}"

            # Use experiment_root here
            model_path = experiment_root / f"PCA_Quantitative_{group_name}" / model_tag
            model_path.mkdir(parents=True, exist_ok=True)

            print(
                f"\n[INFO] Running model: {model_tag} (β={beta}, source={source_ts:.2f}, target={target_ts:.2f})"
            )

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

            structured_results.update(
                {
                    "Model": model_tag,
                    "Beta": beta,
                    "SourceTimestep": source_ts,
                    "TargetTimestep": target_ts,
                }
            )

            model_results.append(structured_results)

        # Save JSON
        group_path = experiment_root / f"PCA_Quantitative_{group_name}"
        group_path.mkdir(exist_ok=True, parents=True)
        out_path = group_path / "group_probe_results.json"
        with out_path.open("w") as f:
            json.dump(model_results, f, indent=2)
        print(f"[INFO] Saved results to: {out_path}")

    # --------------------------------------
    # K-Means PCA Scatter Grid (Fixed)
    # --------------------------------------
    for config_group in all_config_groups:
        group_name = config_group["group_name"]
        configs = config_group["configs"]

        print(f"\n=== Running K-Means visualisation for group: {group_name} ===")

        for config in configs:
            beta = config["beta"]
            checkpoint = config["ckpt"]
            source_ts = config["source_ts"]
            target_ts = config["target_ts"]
            model_tag = config["name"]
            model_name = f"{model_tag}_{dataset_name}"

            # use experiment_root
            project_path = (
                experiment_root / f"PCA_Quantitative_{group_name}" / model_tag
            )
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
