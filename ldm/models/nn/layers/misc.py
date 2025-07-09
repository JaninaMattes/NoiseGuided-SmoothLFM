import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
# Code adapted from:
# - https://github.com/jmtomczak/vae_vampprior/blob/master/utils/nn.py#L40
class NonLinear(nn.Module):
    """ Non-linear layer. """
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h




# Code adapted from: 
# - https://zongyi-li.github.io/blog/2020/fourier-pde/
# - https://github.com/neuraloperator/neuraloperator
class SpectralLinear(nn.Module):    
    """ Linear layer with spectral initialization. """
    def __init__(self, in_features, out_features, bias=True, act_layer=nn.GELU, drop=0.0, frequency_mask='none'):
        super(SpectralLinear, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.frequency_mask = frequency_mask
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
    
    def transform_to_fourier(self, z, freq_threshold=0.5):
        # Perform FFT
        freq_fft = torch.fft.fft(z, dim=-1)
        n_freqs = int(z.shape[-1] * freq_threshold)
        
        if self.frequency_mask == 'low':
            # Mask to zero out high-frequency components
            mask = torch.zeros_like(freq_fft, dtype=torch.bool)
            mask[..., :n_freqs] = True
            mask[..., -n_freqs:] = True
        elif self.frequency_mask == 'high':
            # Mask to zero out low-frequency components
            mask = torch.ones_like(freq_fft, dtype=torch.bool)
            mask[..., :n_freqs] = False
            mask[..., -n_freqs:] = False
        
        freq_fft = freq_fft * mask
        return freq_fft
    
    def forward(self, x):
        h = self.transform_to_fourier(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return h
    
    