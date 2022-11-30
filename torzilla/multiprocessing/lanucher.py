import copy
from torzilla.core import *
from .process import MainProcess, Subprocess
from .manager import Manager


def lanuch(
    num_process=None,
    mainproc=None,
    manager=None,
    subproc=None,
    subproc_args=None,
    rpc=None,
    **shared_args,
):
    # check
    assert_type(num_process, int, null=True)
    assert_type(subproc_args, list, tuple, null=True)
    mainproc_type = _import_module(mainproc, MainProcess)
    subproc_type = _import_module(subproc, Subprocess)
    manager_type = _import_module(manager, Manager)

    if not subproc_args and not num_process:
        raise Exception(f'argument num_process needed')
    elif subproc_args and num_process and len(subproc_args) != num_process:
        raise Exception(f'number of subproc args must be equal to num_process, {len(subproc_args)} != {num_process}')
    elif subproc_args:
        num_process = len(subproc_args)
    
    # check rpc
    has_sub_rpc = False
    if subproc_args is not None:
        for arg in subproc_args:
            if 'rpc' not in arg: continue
            has_sub_rpc = True
            break
    if has_sub_rpc and rpc is not None:
        raise Exception('can not configure subproc rpc and global rpc at same time')

    if has_sub_rpc:
        main_rpc_args = None
        sub_rpc_args = [arg.get('rpc', None) for arg in subproc_args]
    else:
        main_rpc_args, sub_rpc_args = _parse_rpc_args(rpc, num_process)


    # subproc args
    subproc_args = [{} for _ in range(num_process)] if subproc_args is None else subproc_args
    for i, args in enumerate(subproc_args):
        args.update(
            subproc = _import_module(args.get('subproc', subproc_type), Subprocess),
            rpc = sub_rpc_args[i],
            **shared_args
        )

    # mainproc args
    mainproc_args = {}
    mainproc_args.update(dict(
        mainproc = mainproc_type,
        manager = manager_type,
        subproc_args = subproc_args,
        rpc = main_rpc_args,
        **shared_args
    ))

    # start
    proc = mainproc_type(**mainproc_args)
    with proc:
        pass
    return proc


def _parse_rpc_args(rpc_kwargs, num_process):
    if rpc_kwargs is None: return None, [None] * num_process

    enable_mainproc_rpc = rpc_kwargs.get('enable_mainproc_rpc', False)
    rank_start = rpc_kwargs.get('rank', 0)
    world_size = rpc_kwargs.get('world_size', None)

    assert_type(enable_mainproc_rpc, bool)
    assert_type(rank_start, int)
    assert_type(world_size, int, null=True)
    
    if world_size is None:
        world_size = num_process + 1 if enable_mainproc_rpc else num_process
    if world_size <= 0:
        raise ValueError(f'invalud rpc world_size {world_size}')

    rank = 0
    main_args = None
    if enable_mainproc_rpc:
        main_args = copy.copy(rpc_kwargs)
        main_args.update(rank=rank, world_size=world_size)
        rank += 1

    sub_args = [None] * num_process
    for i in range(num_process):
        if rank >= world_size: break
        sub_args[i] = copy.copy(rpc_kwargs)
        sub_args[i].update(rank=rank, world_size=world_size)
        rank += 1
    
    return main_args, sub_args
    


def _import_module(mainproc, dft_type):
    assert_type(mainproc, str, type, null=True)
    if mainproc is None:
        mainproc = dft_type
    elif isinstance(mainproc, str):
        mainproc = import_type(mainproc)
    assert_subclass(mainproc, dft_type)
    return mainproc