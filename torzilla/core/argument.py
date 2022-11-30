import inspect
import argparse

def pick_args(kwargs, keys, drop_none=False, miss_error=False, miss_value=None):
    d = {}
    for k in keys:
        if k not in kwargs and miss_error:
            raise TypeError(f'missing 1 required positional argument: "{k}"')
        v = kwargs.get(k, miss_value)
        if drop_none and v is None: continue
        d[k] = v
    return d


def trace_arg_names(frame=None):

    def parse_code(code):
        # pre process
        code = code.replace('\n', '')

        # check integration
        splits = []
        l, r = code.find('('), None
        pre = l + 1
        num_bracket = 1
        for i in range(pre, len(code), 1):
            if code[i] == '(': 
                num_bracket += 1
            elif code[i] == ')':
                num_bracket -= 1
                if num_bracket == 0:
                    splits.append(code[pre:i].strip())
                    r = i
                    break
            elif code[i] == ',':
                splits.append(code[pre:i].strip())
                pre = i + 1
            else:
                continue
        if r is None: return None

        # parse
        args, kwargs = [], {}
        for s in splits:
            idx = s.find('=')
            if idx < 0:
                if len(kwargs) > 0: raise SyntaxError(f'parse {s} failed, receive a args after kwargs')
                args.append(s)
            else:
                k, v = s[:idx].strip(), s[idx+1:].strip()
                kwargs[k] = v
        return args, kwargs

    # frame
    if frame is None:
        frame = inspect.currentframe()
        call_index = 2
        func_index = 1
    else:
        call_index = 1
        func_index = 0

    frame_infos = inspect.getouterframes(frame)
    
    # call names
    args, kwargs = None, None
    call_frame = frame_infos[call_index].frame
    context, target_index = 1, 0
    while context < 51:
        tb = inspect.getframeinfo(call_frame, context)
        code = ' '.join(tb.code_context[target_index:])
        r = parse_code(code)
        if r is not None:
            args, kwargs = r
            break

        next_tb = inspect.getframeinfo(call_frame, context+1)
        if len(next_tb.code_context) > len(tb.code_context):
            target_index += 1

        context += 2

    if args is None: return {}

    # arg names
    arg_names = inspect.getargvalues(frame_infos[func_index].frame).args
    kwarg_names = {}
    for i, k in enumerate(arg_names):
        if i < len(args):
            kwarg_names[k] = args[i]
        elif k in kwargs:
            kwarg_names[k] = kwargs[k] 
    
    return kwarg_names


def parse_hargs(args, **kwargs):
    from .assertion import assert_type
    assert_type(args, dict, tuple, list)
    
    def _path(root, name):
        return '.'.join([n for n in (root, name) if n and len(n) > 0])

    def dfs_parse(parser, path, args):
        if isinstance(args, Argument):
            name = _path(path, args.name).replace('_', '-')
            parser.add_argument(f'--{name}', **args.kwargs)
            
        elif isinstance(args, (tuple, list)):
            for arg in args:
                dfs_parse(parser, path, arg)
        elif isinstance(args, dict):
            for k, v in args.items():
                next_path = _path(path, k)
                next_parser = parser.add_argument_group(title=k)
                dfs_parse(next_parser, next_path, v)
        else:
            raise TypeError(f'invalid argument type {type(args)} of {path}')

    # parse
    parser = argparse.ArgumentParser(**kwargs)
    dfs_parse(parser, None, args)
    parsed_args = vars(parser.parse_args())

    # set
    out = {}
    for key, value in parsed_args.items():
        root, ks = out, key.split('.')
        for k in ks[:-1]:
            if k not in root: root[k] = {}
            root = root[k]
        root[ks[-1]] = value
    return out



class Argument(object):
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs