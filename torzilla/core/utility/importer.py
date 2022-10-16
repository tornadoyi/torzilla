import importlib
from torzilla.core.error import *
from .assertion import *

def import_module(path): 
    return importlib.import_module(path)

def import_type(path, type=None):
    idx = path.rfind(':')
    if idx < 0: raise ImportError(path, desc='invalid type path format, expected: package:name')
    mod_path, tp_name = path[:idx], path[idx+1:]
    mod = import_module(mod_path)   

    tp = getattr(mod, tp_name)
    if type is not None:
        assert_type(tp, type, name=tp_name)
    return tp