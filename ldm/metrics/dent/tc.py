# Ensure you have BaseMetric and DiagonalGaussianDistribution available
# from .basemetric import BaseMetric # Adjust import
# from ldm.models.nn.dist.distributions import DiagonalGaussianDistribution # Adjust import

import torch
import numpy as np
import math

class TCMetric(BaseMetric):
    """
    Computes Total Correlation (TC) using the batch average estimator from β-TCVAE [1].
    Requires 'full' mode data collection.

    [1] Irina Higgins et al., β-TCVAE: Learning Disentangled Representations From
        Unsupervised Data, ICLR 2018.
    [3] Annette Groth Locatello et al., Challenging Common Assumptions in the
        Unsupervised Learning of Disentangled Representations, PMLR 97:4752-4761, 2019.
    """
    def __init__(self, device: Union[str, torch.device], **kwargs):
        """
        Args:
            device: Device for metric computation.
            **kwargs: Additional keyword arguments (not used by this metric).
        """
        super().__init__(device)
        self._kwargs = kwargs # Optional

    @property
    def _requires(self):
        """
        Requires sampled latent codes (z) and the parameters (mean, log_var)
        of q(z|x) for all data points in the evaluation set.
        """
        return ['latent_samples', 'stats_qzx'] # stats_qzx is [mean, log_var]

    @property
    def _mode(self):
        """TC is a dataset-level metric computed over the full evaluation sample."""
        return 'full'

    def __call__(self, latent_samples: np.ndarray, stats_qzx: List[np.ndarray], **kwargs) -> float:
        """
        Compute Total Correlation (TC) using the batch average estimator.

        Args:
            latent_samples (numpy.ndarray): Samples z ~ q(z|x) for all data points.
                                            Shape (num_samples, latent_dim).
            stats_qzx (list or tuple): List containing mean_qzx [0] and logvar_qzx [1].
                                       Each element is a numpy array (num_samples, latent_dim).

        Returns:
            float: The estimated Total Correlation score.
        """
        # Ensure data is torch tensors on the correct device and appropriate dtype
        z = torch.tensor(latent_samples, dtype=torch.float32, device=self.device)
        means = torch.tensor(stats_qzx[0], dtype=torch.float32, device=self.device)
        log_vars = torch.tensor(stats_qzx[1], dtype=torch.float32, device=self.device)

        num_samples, latent_dim = z.shape

        if num_samples < 2:
             print("Warning: Fewer than 2 samples provided for TC estimation. Returning 0.0.")
             return 0.0

        # Reshape for broadcasting:
        # z_reshaped: (num_samples, 1, latent_dim) - N samples, 1 distribution, D dimensions
        # dist_means/log_vars_reshaped: (1, num_samples, latent_dim) - 1 sample, N distributions, D dimensions
        z_reshaped = z.unsqueeze(1)
        dist_means = means.unsqueeze(0)
        dist_log_vars = log_vars.unsqueeze(0)
        dist_vars = torch.exp(dist_log_vars) # Calculate variances for log PDF

        # Calculate log q(z_j | x_k) for all pairs (j, k)
        # This is the log PDF of N(z | mu_k, sigma^2_k) evaluated at z_j
        # log N(z | mu, sigma^2) = -0.5 * (log(2pi) + log_var) - (z - mu)^2 / (2 * sigma^2)
        # shape: (N_z, N_dist, D) = (num_samples, num_samples, latent_dim)
        # needs torch.tensor(math.pi) 
        log_two_pi = torch.log(torch.tensor(2 * math.pi, device=self.device, dtype=torch.float32))
        
        log_probs_per_dim_all_pairs = -0.5 * (log_two_pi + dist_log_vars) \
                                    - torch.pow(z_reshaped - dist_means, 2) / (2 * dist_vars)

        # Sum over latent dimensions (D) to get log q(z_j | x_k)
        # shape: (num_samples, num_samples)
        log_qz_cond_xi = torch.sum(log_probs_per_dim_all_pairs, dim=2)

        # Estimate log q(z_j) = log (1/N sum_k q(z_j | x_k)) using log-sum-exp trick
        # log (sum exp(a_k)) = logsumexp(a)
        # log (1/N sum q) = log (sum q) - log N = logsumexp(log q) - log N
        # shape: (num_samples,)
        log_qz_estimator = torch.logsumexp(log_qz_cond_xi, dim=1) - math.log(num_samples)

        # Estimate log q(z_ji) = log (1/N sum_k q(z_ji | x_k)) using log-sum-exp trick
        # shape: (N_z, D) = (num_samples, latent_dim)
        log_qzi_estimator = torch.logsumexp(log_probs_per_dim_all_pairs, dim=1) - math.log(num_samples)

        # Compute TC estimator: E_{q(z)} [log q(z) - sum_i log q(z_i)]
        # = (1/N) sum_j [log q(z_j) - sum_i log q(z_ji)]
        # shape: (num_samples,)
        tc_estimator_per_sample = log_qz_estimator - torch.sum(log_qzi_estimator, dim=1)
        tc_score = torch.mean(tc_estimator_per_sample)

        return tc_score.item()
