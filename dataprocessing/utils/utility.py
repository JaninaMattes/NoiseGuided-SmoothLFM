import csv
import gc
import logging
import math
import os

import numpy as np
import torch


################################################################
#                        Utility                        #
################################################################

def round_down_to_nearest_001(value: float) -> float:
    """ Round down the value to the nearest 0.01."""
    return math.floor(value * 100) / 100

def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

def sanitize_filename(filename):
    """Remove parts of filename or PATH that are problematic."""
    return filename.replace('.', '_')




################################################################
#                   Webdatset Utilities                        #
################################################################
def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


def identity(x):
    return x



################################################################
#                 General Torch Utilities                      #
################################################################

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Memory cleaned up.")

def get_device():
    """Get the device (GPU or CPU) for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')






################################################################
#                 Model Setup Utilities                        #
################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)







#############################################################
#               Logging Setup Utilities                     # 
#############################################################
def save_losses_to_csv(losses_dict, filename, log_dir=None):
    """Save losses of training script to a CSV file."""
    try:
        rows = []
        fieldnames = ['epoch', 'train_total_loss', 'val_total_loss', 'train_recon_loss', 'val_recon_loss', 'train_kld_loss', 'val_kld_loss']
        num_epochs = len(losses_dict['train_total_loss'])

        for epoch in range(num_epochs):
            # Store last value per epoch
            row = {
                'epoch': epoch + 1,
                'train_total_loss': losses_dict['train_total_loss'][-1],
                'val_total_loss': losses_dict['val_total_loss'][-1],
                'train_recon_loss': losses_dict['train_recon_loss'][-1],
                'val_recon_loss': losses_dict['val_recon_loss'][-1],
                'train_kld_loss': losses_dict['train_kld_loss'][-1],
                'val_kld_loss': losses_dict['val_kld_loss'][-1]
            }
            rows.append(row)

        # Write data to CSV
        if log_dir:
            filename = os.path.join(log_dir, filename)
                
            with open(filename, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()  # Write column headers
                writer.writerows(rows)  # Write data rows
        
            print(f"Losses saved to {filename}")
        else:
            print("No log directory provided. Not saving losses to CSV.")
    except Exception as e:
        print(f"Error saving losses to CSV: {e}")



def check_image_shape(images):
    """Check images in a list for compatiblity."""
    target_shape = images[0].shape
    for image in images:
        # Find largest width and height
        if image.shape > target_shape:
            target_shape = image.shape
            





if __name__ == "__main__":
    # Test the utility functions
    # Cleanup memory
    sample = torch.randn(1, 4, 32, 32)
    print(sample.shape)
    
    # Normalize 
    norm_sample = normalize_tensor(sample) * 2 - 1
    print(norm_sample.shape)
    print(f"Min: {norm_sample.min()}, Max: {norm_sample.max()}")
    
    # Denormalize
    denorm_sample = denorm_tensor(sample)
    print(denorm_sample.shape)
    print(f"Min: {denorm_sample.min()}, Max: {denorm_sample.max()}")
    