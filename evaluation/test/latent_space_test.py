import os
import sys
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy
from scipy.stats import norm
from torchvision.utils import make_grid
from sklearn.metrics import mutual_info_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)



def get_jointplot_w_marginals(z_embedded):
    """
    Visualise scatter plot and marginal distributions.
    https://stackoverflow.com/questions/65043928/add-a-normal-distribution-to-seaborn-2d-histogram
    """
    df1 = pd.DataFrame(z_embedded, columns=["Latent Dimension 1", "Latent Dimension 2"])
    g = sns.jointplot(
        data=df1, x="Latent Dimension 1", y="Latent Dimension 2",
        marker="+", s=100, marginal_kws=dict(bins=25, fill=False),
    )

    g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

    plt.show()




def traverse_two_latent_dimensions(mu, std, n_samples=16, latent_dim=512, dim_1=0, dim_2=1, digit_size=32):
    """Traverse two latent dimensions."""

    z_dist = torch.distributions.Normal(mu, torch.ones_like(std))
    percentiles = torch.linspace(0.10, 0.9, n_samples)
    grid_x = z_dist.icdf(percentiles[:, None].repeat(1, latent_dim))
    grid_y = z_dist.icdf(percentiles[:, None].repeat(1, latent_dim))

    figure = np.zeros((digit_size * n_samples, digit_size * n_samples, 3))  # Initialize with 3 channels for RGB
    z_sample_default = mu.clone().detach().cpu()

    samples = []
    for yi in range(n_samples):
        for xi in range(n_samples):
            z_sample = z_sample_default.clone()
            z_sample[:, dim_1] = grid_x[xi, dim_1]
            z_sample[:, dim_2] = grid_y[yi, dim_2]

            sample = z_sample.view(1, latent_dim)
            samples.append(sample)
        
    return samples



def get_class_means(latents_mean, latents_std, labels, latent_dim=512, n_samples=16):
    classes_mean = {}
    unique_labels = np.unique(labels.cpu().numpy())

    for label in unique_labels:
        class_mean = latents_mean[labels == label].mean(dim=0, keepdims=True)
        class_std = latents_std[labels == label].mean(dim=0, keepdims=True)
        classes_mean[label] = (class_mean, class_std)
    
    samples = []
    # Visualize class means
    for label, (class_mean, class_std) in classes_mean.items():
        # Create normal distribution of current class
        class_dist = torch.distributions.Normal(class_mean, class_std)
        percentiles = torch.linspace(0.05, 0.95, steps=n_samples)
        # get samples from different parts of the distribution using icdf
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.icdf 

        class_z_samples = class_dist.icdf(percentiles[:, None].repeat(1, latent_dim))

        print(f"Class {label}: {class_z_samples.shape}")
        
        samples.append(class_z_samples)
    
    return classes_mean, samples



def get_correlation_matrix(latents):
    """Visualize correlation matrix of latent space."""
    
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()  
    
    corr_matrix = torch.corrcoef(latents.T)
    
    # Visualize correlation matrix
    corr_matrix = corr_matrix.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False)
    plt.title('Correlation Matrix of Latent Space', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.close(fig)




def compute_disentanglement_metrics(latents):
    metrics = {}

    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()

    def discretize(values, bins=10):
        return np.digitize(values, np.linspace(values.min(), values.max(), bins))

    # Modularity Score
    def modularity_score(latents):
        """
        Measure how independent latent dimensions are.
        A lower score indicates better disentanglement.
        """
        try:
            corr_matrix = np.corrcoef(latents.T)
            return np.mean(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
        except Exception as e:
            print(f"Error in modularity_score: {e}")
            raise

    # Entropy of Latent Dimensions
    def latent_entropy(latents):
        """
        Measure the entropy of each latent dimension.
        Higher entropy indicates more information captured.
        """
        try:
            return np.mean([entropy(discretize(latent)) for latent in latents.T])
        except Exception as e:
            print(f"Error in latent_entropy: {e}")
            raise

    # Mutual Information Score
    def mutual_information(latents):
        """
        Measure the mutual information between latent dimensions.
        Lower mutual information indicates better disentanglement.
        """
        try:
            mi_scores = []
            for i in range(latents.shape[1]):
                for j in range(i + 1, latents.shape[1]):
                    mi_scores.append(mutual_info_score(discretize(latents[:, i]), discretize(latents[:, j])))
            return np.mean(mi_scores)
        except Exception as e:
            print(f"Error in mutual_information: {e}")
            raise

    # Compactness Score
    def compactness_score(latents):
        """
        Measure the compactness of the latent space.
        Lower compactness indicates better disentanglement.
        """
        try:
            return np.mean(np.var(latents, axis=0))
        except Exception as e:
            print(f"Error in compactness_score: {e}")
            raise

    try:
        metrics = {
            'modularity': modularity_score(latents),
            'latent_entropy': latent_entropy(latents),
            'mutual_information': mutual_information(latents),
            'compactness': compactness_score(latents)
        }
    except Exception as e:
        print(f"Error computing disentanglement metrics: {e}")

    return metrics




def test():
    latents = torch.randn(128, 4, 32, 32)               # (128, 4, 32, 32)
    latent_vectors = torch.randn(128, 512)              # (128, 512)
    logvar_vectors = torch.randn(128, 512)              # (128, 512)
    mean_vectors = torch.randn(128, 512)                # (128, 512)
    std_vectors = torch.exp(0.5 * logvar_vectors)       # (128, 512)
    labels = torch.randint(0, 10, (128,))               # (128,)
    latent_embedded = torch.randn(128, 2)               # (128, 2)


    # # Compute disentanglement metrics
    # metrics = compute_disentanglement_metrics(latent_vectors)
    # print(metrics)


    # Compute class means
    class_mean_samples, generated_samples = get_class_means(
        mean_vectors, std_vectors, labels, latent_dim=512
    )
    # print(samples)

    selected_label = list(class_mean_samples.keys())[0]

    mu, std = class_mean_samples[selected_label]
    latent_dim = latent_vectors.shape[1]
    
    traversed_sampel = traverse_two_latent_dimensions(
        mu, std, latent_dim=latent_dim, dim_1=0, dim_2=1, digit_size=32)
    
    print(len(traversed_sampel))
    print(f"Shape: {traversed_sampel[0].shape}")


if __name__ == "__main__":
    test()
