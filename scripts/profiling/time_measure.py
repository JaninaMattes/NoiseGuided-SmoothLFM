# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#best-practices-and-recommendations

import torch 

def timed(fn):
    """ Returns the result of running `fn()` and the time it took for `fn()` to run,
        in seconds. We use CUDA events and synchronization for the most accurate measurements.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000