import random
import numpy as np

import torch
import torchvision.transforms as transforms


# ------------------------------------------------------------------------------
# Custom transforms
# ------------------------------------------------------------------------------
class CustomRandHorizontalFlip:
    """Randomly flips image and latent tensors horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if torch.rand(1) < self.prob:
            image, latent = sample['image'], sample['latent']
            image = transforms.functional.hflip(image)
            latent = transforms.functional.hflip(latent)
            sample['image'], sample['latent'] = image, latent
        return sample


class CustomMultiRandHorizontalFlip:
    """Randomly flips image and latent tensors horizontally, excluding 'label'."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if torch.rand(1) < self.prob:
            for key, value in sample.items():
                if key != 'label': # Exclude 'label'
                    sample[key] = transforms.functional.hflip(value)
        return sample


class CustomNormalization:
    """Normalizes latent tensor to [-1, 1] range."""

    def __init__(self, timestep=0.0, epsilon=1e-8):
        self.timestep = timestep
        self.epsilon = epsilon  # Avoid division by zero

    def __call__(self, sample):
        # Only for pixel-based datasets
        if self.timestep > 1.0:
            sample = sample['image']
            sample_min, sample_max = torch.aminmax(sample)

            if (sample_max - sample_min) > self.epsilon:
                sample = (sample - sample_min) / (sample_max - sample_min)  # Normalize to [0,1]
                sample = sample * 2.0 - 1.0                                 # Scale to [-1,1]

            sample.update({'image': sample})

        return sample