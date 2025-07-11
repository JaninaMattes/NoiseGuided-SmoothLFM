# Code adapted from: Variational Learning with Disentanglement-PyTorch
# https://github.com/google-research/disentanglement_lib
# https://github.com/amir-abdi/disentanglement-pytorch
import math
from typing import Union

import torch
from torch.nn import functional as F
import torch.nn as nn

from jutils import instantiate_from_config
from jutils import exists, freeze, default

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.ignite.metrics import MutualInformation
from torchmetrics.clustering import MutualInfoScore

# other imports
from ldm.models.architecture.base_architectures import BaseDecoder, BaseEncoder, BaseVAE
from ldm.models.nn.dist.distributions import DiagonalGaussianDistribution
from ldm.models.nn.out.outputs import BetaVAEModelOutput, DecoderOutput, EncoderOutput



class VAEMetricsTracker(nn.Module):
    """ Evaluate the learned representations of a VAE model.
        [1] Locatello et al., 2019 "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
        [2] BetaVAE (Higgins et al., 2017) "Learning basic visual concepts with a constrained variational framework."
        [3] FactorVAE (Kim and Mnih, 2018) "Disentangling by factorising."
        [4] Mutual Information Gap (MIG) (Chen et al., 2018) "Isolating sources of disentanglement in variational autoencoders."
        [5] Interventional Robustness Score (IRS) (Suter et al., 2019) "Robustly disentangled causal mechanisms: Validating deep representations for interventional robustness."
        [6] Disentanglement Completeness and Informativeness (DCP) (Eastwood and Williams, 2018) "A framework for the quantitative evaluation of disentangled representations."
        [7] Separated Attribute Predictability (SAP) (Kumar et al., 2018). "Variational inference of disentangled latent concepts from unlabeled observations."
        
    """
    def __init__(self, n_latent, n_factors, n_attributes=None):
        super().__init__()
        self.mutual_info = MutualInformation()
        self.mig = MIG(device=device, n_latent=n_latent, n_factors=n_factors)
        self.sap = SAPd(device=device, n_latent=n_latent, n_factors=n_factors)
        self.dcp = DCP(device=device, n_latent=n_latent, n_factors=n_factors)
        self.reset()
    
    
    def __call__(self, target, pred):
        pass 
        
    def aggregate(self):
        mig_score = torch.stack(self.mig_scores).mean()
        # Compute the average of the disentanglement scores
        disentanglement_score = torch.stack(self.disentanglement_scores).mean()
        factor_vae_score = torch.stack(self.factor_vae_scores).mean()
        beta_tcvae_score = torch.stack(self.beta_tcvae_scores).mean()
        modularity_score = torch.stack(self.modularity_scores).mean()
        sap_score = torch.stack(self.sap_scores).mean()
        completeness_score = torch.stack(self.completeness_scores).mean()
        compactness_pca_variance = torch.stack(self.compactness_pca_variance).mean()
        completeness_accuracy = torch.stack(self.completeness_accuracy).mean()
        return dict(
            mig_score=mig_score,
            disentanglement_score=disentanglement_score,
            factor_vae_score=factor_vae_score,
            beta_tcvae_score=beta_tcvae_score,
            modularity_score=modularity_score,
            sap_score=sap_score,
            completeness_score=completeness_score,
            compactness_pca_variance=compactness_pca_variance,
            completeness_accuracy=completeness_accuracy
        )


    def reset(self):
        self.mig_scores = []
        self.disentanglement_scores = []
        self.factor_vae_scores = []
        self.beta_tcvae_scores = []
        self.modularity_scores = []
        self.sap_scores = []
        self.completeness_scores = []
        self.compactness_pca_variance = []
        self.completeness_accuracy = []