import torch
import torch as th
import torch.nn as nn


class ScaledTanh(nn.Module):
    """Scaled tanh activation function."""
    def __init__(self, init_scale=3.0):
        super().__init__()
        self.scale_log = nn.Parameter(th.log(torch.tensor(init_scale)), requires_grad=True)
    
    @property
    def scale(self):
        return self.scale_log.exp()
    
    def forward(self, x):
        return torch.tanh(x) * self.scale




if __name__ == "__main__":
    # Test scaled tanh
    stanh = ScaledTanh(4.0)
    z = (1, 512)
    x = torch.normal(0, 1, size=z)    
    y = stanh(x)
    