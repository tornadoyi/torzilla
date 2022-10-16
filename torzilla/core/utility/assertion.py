from torzilla.core.error import *
from .argument import trace_arg_names

def assert_type(arg, *types, strict=False, null=False, name=None):
    if null and arg is None: return
    if strict:
        tp = type(arg)
        for _, t in enumerate(types):
            if tp == t: return
        name = name or trace_arg_names().get('name', 'unknown argument name')
        raise ArgumentTypeError(name, arg, types)
    else:
        if isinstance(arg, tuple(types)): return
        raise ArgumentTypeError(name, arg, types)


def assert_subclass(child, parent, name=None):
    if issubclass(child, parent): return
    name = name or trace_arg_names().get('name', 'unknown argument name')
    raise NotSubclassError(name, child, parent)