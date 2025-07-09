# Implementation of nested dropout (https://arxiv.org/abs/1402.0915)
# Generates an ordered code for the latent space of the encoder output
import random
import torch
import torch.nn as nn


""" Nested Dropout."""

class NestedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, k_keep=None):
        """ 
        Randomly drop encoded tokens in a nested manner during training 
        to generate an ordered latent code.
        
        Args:
            tokens (torch.Tensor): Encoded tokens from the encoder.
                Shape: (B, S), where S = Sequence Length.
        """
        B, T, D = x.shape                                                   # B: batch size, T: num tokens, D: embedding dimensionality
        
        if k_keep is None:
            k_keep = torch.randint(1, T + 1, (B,), device=x.device)         # Different length per batch
            k_keep = torch.clamp(k_keep, 1, T)                              # Ensure 1 ≤ k_keep ≤ T

        batch_indices = torch.arange(T, device=x.device).expand(B, -1)      # Shape: [B, T]
        mask = batch_indices < k_keep.unsqueeze(1)                          # Shape: [B, T]

        return x * mask.unsqueeze(-1)                                       # Shape: [B, T, D], Masked tokens







if __name__ == '__main__':
    # Test NestedDropout
    dropout = NestedDropout()
    tokens = torch.randn(1, 256)  # (Batch, Sequence_len)
    
    print("Original Tokens Shape:", tokens.shape)
    print(f"Original Tokens:\n{tokens}")

    masked_tokens, mask = dropout(tokens)
    print("Dropped Tokens Shape:", masked_tokens.shape)
    print("Mask Shape:", mask.shape)
    print(f"Dropped Tokens:\n{masked_tokens}")
    print(f"Mask:\n{mask}")
