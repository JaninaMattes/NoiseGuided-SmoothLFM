import csv
import gc
import logging
import math
import os

import numpy as np
import torch





################################################################
#                        Normalization                        #
################################################################

def z_score_normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """
    Perform z-score normalization on a tensor.

    Args:
        tensor (torch.Tensor): The input tensor to normalize.
        dim (int): The dimension along which to compute the mean and std. Default is 0.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True) + eps # avoid division by zero

    return (tensor - mean) / std


def normalize_minus_one_one_np(x: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to the range [-1, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_norm.astype(np.float32)

def normalize_zero_one_np(x: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to the range [0, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm.astype(np.float32)


def normalize_zero_one_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalize a PyTorch tensor to the range [0, 1]."""
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm.float()


def normalize_minus_one_one_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalize a PyTorch tensor to the range [-1, 1]."""
    x_min = x.min()
    x_max = x.max()
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_norm.float()


def normalize_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the range [-1, 1]."""
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x * 2 - 1
    return x.to(torch.float32)


def normalize_zero_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the range [0, 1]."""
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    return x


def normalize_minus_one_one(tensor:torch.Tensor, min_val:float=None, max_val:float=None, eps:float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to the range [-1, 1]."""
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    
    if max_val - min_val > 0:
        tensor = 2 * (tensor - min_val) / (max_val - min_val + eps) - 1
        tensor = torch.clamp(tensor, -1, 1).to(torch.float32)
    else:
        tensor = torch.zeros_like(tensor) # If constant, return zeros
    
    return tensor


def normalize_zero_one(tensor:torch.Tensor, min_val:float=None, max_val:float=None, eps:float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to the range [0, 1]."""
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    
    if max_val - min_val > 0:
        tensor = (tensor - min_val) / (max_val - min_val + eps)
        tensor = torch.clamp(tensor, 0, 1).to(torch.float32)
    else:
        tensor = torch.zeros_like(tensor) # If constant, return zeros
    
    return tensor


def normalize_minus_one_one_np(array: np.ndarray, min_val:float=None, max_val:float=None, eps: float = 1e-8) -> np.ndarray:
    """Normalize a NumPy array to the range [-1, 1]."""
    if min_val is None:
        min_val = np.min(array)
    if max_val is None:
        max_val = np.max(array)
    
    if max_val - min_val > 0:
        array = 2 * (array - min_val) / (max_val - min_val + eps) - 1
        array = np.clip(array, -1, 1).astype(np.float32)
    else:
        array = np.zeros_like(array, dtype=np.float32)  # If constant, return zeros
    
    return array


def normalize_zero_one_np(array: np.ndarray, min_val:float=None, max_val:float=None, eps: float = 1e-8) -> np.ndarray:
    """Normalize a NumPy array to the range [0, 1]."""
    if min_val is None:
        min_val = np.min(array)
    if max_val is None:
        max_val = np.max(array)
    
    if max_val - min_val > 0:
        array = (array - min_val) / (max_val - min_val + eps)
        array = np.clip(array, 0, 1).astype(np.float32)
    else:
        array = np.zeros_like(array, dtype=np.float32)  # If constant, return zeros
    
    return array


def denorm_tensor(tensor, min=0.0, max=255.0, keep_channels=3):
    """
    Denormalize a tensor to [min, max] range for visualization.

    Args:
        tensor (torch.Tensor): Shape (B, C, H, W) or (C, H, W)
        min (float): Minimum output value
        max (float): Maximum output value
        keep_channels (int): Trim to N channels (e.g., 3 for RGB)

    Returns:
        torch.Tensor: Uint8 tensor scaled to [min, max]
    """
    tensor = tensor.to(torch.float32)

    # Handle single image (C, H, W) 
    is_batched = tensor.dim() == 4
    if not is_batched:
        tensor = tensor.unsqueeze(0)  # shape becomes (1, C, H, W)

    # Ensure tensor has at least 3 channels
    if tensor.size(1) > keep_channels:
        tensor = tensor[:, :keep_channels]

    orig_min = tensor.amin(dim=(2, 3), keepdim=True)
    orig_max = tensor.amax(dim=(2, 3), keepdim=True)

    scale = (max - min) / (orig_max - orig_min + 1e-8)
    offset = min - orig_min * scale

    x = tensor * scale + offset
    x = torch.clamp(x, min, max).round().to(torch.uint8)

    return x if is_batched else x[0]  # remove batch dim if input was unbatched




def denorm_metrics_tensor(tensor, target_range=(0, 1), dtype='float'):
    """
    Automatically scales a batch of tensors from its global min/max range to a target range.

    Args:
        tensor: Tensor of shape (B, C, H, W).
        target_range: Tuple (min, max) for the output range. E.g., (0, 1) or (0, 255).
        dtype: 'float' → returns float32 tensor (recommended for metrics like LPIPS, SSIM).
               'int' → returns uint8 tensor (e.g., for saving or FID if needed).

    Returns:
        Tensor scaled to target_range, converted to correct dtype.
    """
    tensor = tensor.to(torch.float32)

    orig_min = tensor.amin()
    orig_max = tensor.amax()

    # Handle case where min == max
    if orig_max > orig_min:
        scale = (target_range[1] - target_range[0]) / (orig_max - orig_min)
        offset = target_range[0] - orig_min * scale
        tensor_scaled = tensor * scale + offset
    else:
        # Fill with mid-value if constant tensor
        tensor_scaled = torch.full_like(tensor, (target_range[0] + target_range[1]) / 2)

    # Clamp to avoid numerical overshoots
    tensor_scaled = torch.clamp(tensor_scaled, target_range[0], target_range[1])

    # Handle dtype conversion
    if dtype == 'float':
        tensor_scaled = tensor_scaled.to(torch.float32)
    elif dtype == 'int':
        # If target range is (0, 1), first scale to (0, 255)
        if target_range[1] <= 1.0:
            tensor_scaled = tensor_scaled * 255.0
        tensor_scaled = torch.round(torch.clamp(tensor_scaled, 0, 255)).to(torch.uint8)
    else:
        raise ValueError("dtype must be 'float' or 'int'.")

    return tensor_scaled




def normalize_tensor(tensor, target_min=0, target_max=1, dtype='float') -> torch.Tensor:
    """Normalize a tensor tensor with multiple channels for visualization."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    # Output tensor
    normalized = torch.zeros_like(tensor, dtype=torch.float32)
    
    # Iterate over the batch
    for i in range(tensor.size(0)):
        img = tensor[i]
        orig_min = img.min().item()
        orig_max = img.max().item()
        
        # Skip if image is constant
        if orig_max == orig_min:
            normalized[i] = torch.zeros_like(img)
            continue
            
        # Normalize to [0,1] and scale to range
        img_normalized = (img - orig_min) / (orig_max - orig_min)
        img_normalized = img_normalized * (target_max - target_min) + target_min
        normalized[i] = img_normalized
    
    if dtype == 'float':
        normalized = normalized.to(torch.float32)
    elif dtype == 'int':
        if target_max <= 1:
            normalized = normalized * 255
        normalized = torch.clamp(torch.round(normalized), 0, 255)
        normalized = normalized.to(torch.uint8)
    
    return normalized






if __name__ == "__main__":
    
    print("Testing normalization functions...")
    B, C, H, W = 16, 3, 256, 256
    random_tensor = torch.randn(B, C, H, W) * 2.5 - 1.2  # Range ~[-12, 9]
    
    print(f"Min: {random_tensor.min():.4f}, Max: {random_tensor.max():.4f}, Shape: {random_tensor.shape}, Dtype: {random_tensor.dtype}")

    tensor_float = denorm_metrics_tensor(random_tensor, target_range=(0, 1), dtype='float')
    print(f"Float range: min={tensor_float.min():.4f}, max={tensor_float.max():.4f}, dtype={tensor_float.dtype}")
    
    tensor_uint8 = denorm_metrics_tensor(random_tensor, target_range=(0, 1), dtype='int')
    print(f"Uint8 range: min={tensor_uint8.min()}, max={tensor_uint8.max()}, dtype={tensor_uint8.dtype}")

    assert 0.0 <= tensor_float.min() and tensor_float.max() <= 1.0, "Float scaling failed!"
    assert tensor_uint8.min() >= 0 and tensor_uint8.max() <= 255, "Uint8 scaling failed!"


    # Change target range to [-1, 1]
    tensor_float = denorm_metrics_tensor(random_tensor, target_range=(-1, 1), dtype='float')
    print(f"Float range [-1, 1]: min={tensor_float.min():.4f}, max={tensor_float.max():.4f}, dtype={tensor_float.dtype}")
    
    tensor_uint8 = denorm_metrics_tensor(random_tensor, target_range=(0, 255), dtype='int')
    print(f"Uint8 range [0, 255]: min={tensor_uint8.min()}, max={tensor_uint8.max()}, dtype={tensor_uint8.dtype}")
    
    assert -1.0 <= tensor_float.min() and tensor_float.max() <= 1.0, "Float scaling to [-1, 1] failed!"
    assert tensor_uint8.min() >= 0 and tensor_uint8.max() <= 255, "Uint8 scaling to [-1, 1] failed!"


    print("Testing normalization functions on latent tensor...")
    B, C, H, W = 16, 4, 32, 32
    random_tensor = torch.randn(B, C, H, W) * 2.5 - 1.2  # Range ~[-12, 9]
    print(f"Min: {random_tensor.min():.4f}, Max: {random_tensor.max():.4f}, Shape: {random_tensor.shape}, Dtype: {random_tensor.dtype}")

    tensor_float = denorm_metrics_tensor(random_tensor, target_range=(0, 1), dtype='float')
    print(f"Float range: min={tensor_float.min():.4f}, max={tensor_float.max():.4f}, dtype={tensor_float.dtype}")
    
    tensor_uint8 = denorm_metrics_tensor(random_tensor, target_range=(0, 1), dtype='int')
    print(f"Uint8 range: min={tensor_uint8.min()}, max={tensor_uint8.max()}, dtype={tensor_uint8.dtype}")

    assert 0.0 <= tensor_float.min() and tensor_float.max() <= 1.0, "Float scaling failed!"
    assert tensor_uint8.min() >= 0 and tensor_uint8.max() <= 255, "Uint8 scaling failed!"


    # Change target range to [-1, 1]
    tensor_float = denorm_metrics_tensor(random_tensor, target_range=(-1, 1), dtype='float')
    print(f"Float range [-1, 1]: min={tensor_float.min():.4f}, max={tensor_float.max():.4f}, dtype={tensor_float.dtype}")
    
    tensor_uint8 = denorm_metrics_tensor(random_tensor, target_range=(0, 255), dtype='int')
    print(f"Uint8 range [0, 255]: min={tensor_uint8.min()}, max={tensor_uint8.max()}, dtype={tensor_uint8.dtype}")
    
    assert -1.0 <= tensor_float.min() and tensor_float.max() <= 1.0, "Float scaling to [-1, 1] failed!"
    assert tensor_uint8.min() >= 0 and tensor_uint8.max() <= 255, "Uint8 scaling to [-1, 1] failed!"
    

    print("Latent tensor checks passed!")
    
    

    print("\nTesting normalization functions with different ranges...")
    # Example usage
    ipt1 = torch.randn(4, 4) * 10  # Random tensor
    ipt2 = torch.full((4, 4), 5)   # Constant tensor

    # Normalize to [-1, 1]
    normalized_minus_one_one = normalize_minus_one_one(ipt1)
    print("Normalized to [-1, 1]:\n", normalized_minus_one_one)

    # Normalize to [0, 1]
    normalized_zero_one = normalize_zero_one(ipt1)
    print("Normalized to [0, 1]:\n", normalized_zero_one)

    # Handle constant tensor normalization
    normalized_constant = normalize_zero_one(ipt2)
    print("Normalized constant tensor to [0, 1]:\n", normalized_constant)