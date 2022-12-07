from .argument import trace_arg_names

def assert_type(arg, *types, strict=False, null=False, name=None):
    if null and arg is None: return
    if strict:
        tp = type(arg)
        for _, t in enumerate(types):
            if tp == t: return
        name = name or trace_arg_names().get('arg', 'unknown')
        raise TypeError(f'type of "{name}" is {type(arg)}, expect: {", ".join([tp.__name__ for tp in types])}')
    else:
        if isinstance(arg, tuple(types)): return
        name = name or trace_arg_names().get('arg', 'unknown')
        raise TypeError(f'Type of "{name}" is {type(arg)}, expect: {", ".join([tp.__name__ for tp in types])}')


def assert_subclass(child, parent, name=None):
    if issubclass(child, parent): return
    name = name or trace_arg_names().get('child', 'unknown argument name')
    raise TypeError(f'type {child} ({name}) is not subclass of {parent}')