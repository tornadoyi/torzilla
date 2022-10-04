from torzilla.core.error import *

def assert_type(name, arg, *types, strict=False, null=False):
    if null and arg is None: return
    if strict:
        tp = type(arg)
        for _, t in enumerate(types):
            if tp == t: return
        raise ArgumentTypeError(name, arg, types)
    else:
        if isinstance(arg, tuple(types)): return
        raise ArgumentTypeError(name, arg, types)


def assert_subclass(name, child, parent):
    if issubclass(child, parent): return
    raise NotSubclassError(name, child, parent)