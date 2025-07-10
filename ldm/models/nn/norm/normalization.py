import torch
import torch as th
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
    
def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    
    taken from: https://github.com/phizaz/diffae/blob/master/model/nn.py#L99
    """
    return GroupNorm32(min(32, channels), channels)