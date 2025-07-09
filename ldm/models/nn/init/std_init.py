import torch 
import torch.nn as nn
import torch.nn.init as init


""" Standard Initializations """


def initialize_weights(
    model: nn.Module,
    init_type: str = 'kaiming',
    **kwargs
) -> None:
    """Initialize network weights.
    
    Args:
        init_type: initialization method: 'normal' | 'xavier' | 'kaiming' (default)
        **kwargs: additional arguments passed to the initialization method
    """
    init_funcs = {
        'normal': normal_init,
        'xavier': xavier_init,
        'kaiming': kaiming_init
    }
    
    if init_type not in init_funcs:
        raise ValueError(f"Initialization method {init_type} not supported. "
                       f"Supported methods are: {list(init_funcs.keys())}")
    
    init_func = init_funcs[init_type]
    model.apply(lambda x: init_func(x, **kwargs))
    
    
    
def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
    bias: bool = True
) -> None:
    """Initialize network weights using Kaiming initialization.
    
    Args:
        a: negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of 
              the variance of the weights in the forward pass. Choosing 'fan_out' preserves the 
              magnitudes in the backwards pass.
        nonlinearity: the non-linear function (nn.functional name), recommended to use only with 
                     'relu' or 'leaky_relu' (default).
        bias: whether to also initialize the bias (if it exists)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(
            module.weight,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity
        )
        if bias and module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.constant_(module.weight, 1)
        if module.bias is not None:
            init.constant_(module.bias, 0)


def normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    bias: bool = True
) -> None:
    """Initialize network weights using Normal distribution.
    
    Args:
        mean: mean of the normal distribution
        std: standard deviation of the normal distribution
        bias: whether to also initialize the bias (if it exists)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.normal_(module.weight, mean=mean, std=std)
        if bias and module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.constant_(module.weight, 1)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: bool = True
) -> None:
    """Initialize network weights using Xavier initialization.
    
    Args:
        gain: gain factor to apply (default: 1.0)
        bias: whether to also initialize the bias (if it exists)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.xavier_normal_(module.weight, gain=gain)
        if bias and module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.constant_(module.weight, 1)
        if module.bias is not None:
            init.constant_(module.bias, 0)
