"""
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright 2018 The DisentanglementLib Authors.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

ORIGINAL CODE WAS CHANGED AS FOLLOWS:
- Conversion from Tensorflow to PyTorch.
- Integration as a mergable BaseMetric that can be combined with multiple other metrics for efficient computation.
- Efficiency improvements through parallelization.
- Function and variable renaming.
"""
from fastargs.decorators import param
import numpy as np
import sklearn.preprocessing
from sklearn import svm
import torch

from .basemetric import BaseMetric

# --- SAPd Metric (Example 'full' mode metric) ---
# Requires gt_factors and mean of q(z|x) (passed as stats_qzx[0])
class SAPd(BaseMetric):
    def __init__(self, device, num_train=10000, num_test=5000, num_bins=20, **kwargs):
        super().__init__(device)
        self.num_train = num_train
        self.num_test = num_test
        self.num_bins = num_bins
        self._kwargs = kwargs

    @property
    def _requires(self):
        """
        Requires [mean of q(z|x), logvar of q(z|x)] (as 'stats_qzx')
        and ground truth factors ('gt_factors').
        """
        return ['stats_qzx', 'gt_factors']

    @property
    def _mode(self):
        """SAP is computed over the full dataset/sample."""
        return 'full'

    def __call__(self, stats_qzx: List[np.ndarray], gt_factors: np.ndarray, **kwargs) -> float:
        """
        Compute the Separated Attribute Predictability Score [1].

        Args:
            stats_qzx (list or tuple): List containing mean_qzx [0] and logvar_qzx [1].
                                       Each element is a numpy array (num_samples, latent_dim).
            gt_factors (numpy.ndarray): Ground truth factors (num_samples, num_factors).

        Returns:
            float: The computed SAP score.
        """
        mean_qzx = stats_qzx[0] # SAPd implementation specifically uses the mean

        if isinstance(mean_qzx, torch.Tensor): mean_qzx = mean_qzx.detach().cpu().numpy()
        if isinstance(gt_factors, torch.Tensor): gt_factors = gt_factors.detach().cpu().numpy()

        num_total_samples = len(mean_qzx)
        min_required_samples = self.num_train + self.num_test

        if num_total_samples < min_required_samples:
             raise ValueError(
                f'SAPd requires at least {min_required_samples} samples for train/test splits, but received {num_total_samples}.'
            )

        # Use a fixed random seed for reproducible train/test splits for the metric calculation itself
        original_rng_state = np.random.get_state()
        np.random.seed(42) # Use a fixed seed


        total_idcs = np.arange(num_total_samples)
        train_idcs = np.random.choice(total_idcs, self.num_train, replace=False)
        test_idcs = np.setdiff1d(total_idcs, train_idcs)

        if len(test_idcs) < self.num_test:
             print(f"Warning: Fewer test samples available ({len(test_idcs)}) than requested ({self.num_test}) for SAPd test split. Using all available.")
             test_idcs = test_idcs
        else:
             test_idcs = np.random.choice(test_idcs, self.num_test, replace=False)

        np.random.set_state(original_rng_state) # Restore original rng state


        # Scale data to [0, 1] before binning/SVC using the full dataset range
        mean_qzx_scaled = sklearn.preprocessing.minmax_scale(mean_qzx, axis=0)
        gt_factors_scaled = sklearn.preprocessing.minmax_scale(gt_factors, axis=0)

        # Bin the scaled data
        bins = np.linspace(0.0, 1.0 + 1e-8, self.num_bins + 1)
        mean_qzx_binned = np.digitize(mean_qzx_scaled, bins[:-1], right=False).astype(int) -1
        gt_factors_binned = np.digitize(gt_factors_scaled, bins[:-1], right=False).astype(int) -1

        # Clamp bin indices to be within the valid range [0, num_bins - 1]
        mean_qzx_binned = np.clip(mean_qzx_binned, 0, self.num_bins - 1)
        gt_factors_binned = np.clip(gt_factors_binned, 0, self.num_bins - 1)


        mean_qzx_train = mean_qzx_scaled[train_idcs]
        gt_factors_train = gt_factors_binned[train_idcs]
        mean_qzx_test = mean_qzx_scaled[test_idcs]
        gt_factors_test = gt_factors_binned[test_idcs]

        num_latents = mean_qzx.shape[-1]
        num_factors = gt_factors.shape[-1]
        scores = np.zeros([num_latents, num_factors]) # scores[i, j] is accuracy of latent i predicting factor j

        for i in range(num_latents):
            for j in range(num_factors):
                mu_train_i = mean_qzx_train[:, i]
                gt_factor_train_j = gt_factors_train[:, j]

                mu_test_i = mean_qzx_test[:, i]
                gt_factor_test_j = gt_factors_test[:, j]

                classifier = svm.LinearSVC(C=0.01, class_weight='balanced', random_state=42, dual="auto")

                # Check if the target factor j has more than one unique value in the training set
                unique_train_labels = np.unique(gt_factor_train_j)
                if len(unique_train_labels) < 2:
                    scores[i, j] = 0.0
                    # print(f"Warning: GT factor {j} constant in training subset. Cannot train SVC.") # Optional verbose warning
                else:
                    classifier.fit(mu_train_i[:, np.newaxis], gt_factor_train_j)
                    pred = classifier.predict(mu_test_i[:, np.newaxis])
                    scores[i, j] = np.mean(pred == gt_factor_test_j)

        # Compute SAP score: Mean difference between top 1 and top 2 accuracy for each factor
        sorted_scores = np.sort(scores, axis=0)
        sap_score = np.mean(sorted_scores[-1, :] - sorted_scores[-2, :])

        return sap_score # Return a single scalar score
