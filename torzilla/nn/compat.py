import torch
if float('.'.join(torch.__version__.split('.')[:2])) < 1.9:
    from functools import partial
    from torch import nn
    def compat_device_dtype(f, *args, device=None, dtype=None, **kwargs):
        return f(*args, **kwargs)
    nn.Linear = partial(compat_device_dtype, nn.Linear)
    nn.LayerNorm = partial(compat_device_dtype, nn.LayerNorm)
    nn.BatchNorm1d = partial(compat_device_dtype, nn.BatchNorm1d)