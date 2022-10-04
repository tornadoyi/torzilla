
from torzilla.core.error import *

def pick_args(kwargs, keys, drop_none=False, miss_error=False, miss_value=None):
    d = {}
    for k in keys:
        if k not in kwargs and miss_error:
            raise ArgumentMissError(k, desc='pick_args')
        v = kwargs.get(k, miss_value)
        if drop_none and v is None: continue
        d[k] = v
    return d