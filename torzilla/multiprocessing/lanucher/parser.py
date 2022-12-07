import copy
from torzilla.core import import_type, assert_subclass, assert_type
from torzilla.multiprocessing.target import Target
from torzilla.multiprocessing.manager import Manager


def parse(
    num_process=None,
    main=None,
    manager=None,
    target=None,
    args=None,
    rpc=None,
    **shared_args,
):
    # check
    assert_type(num_process, int, null=True)
    assert_type(args, list, tuple, null=True)
    assert_type(rpc, dict, null=True)
    main = _import_target(main, Target, name='main')
    target = _import_target(target, Target, name='target')
    manager = _import_manager(manager)

    # num process
    if num_process is None and args is None:
        raise ValueError('num_process or args should be set')
    elif (
        num_process is not None and 
        args is not None and
        len(args) != num_process
    ):
        raise ValueError(f'num_process != len(args), {num_process} != {len(args)}')
    elif args is not None:
        num_process = len(args)

    # check args
    args = [{} for _ in range(num_process)] if args is None else [copy.copy(arg) for arg in args]
    for arg in args:
        assert_type(arg, dict)
        arg['target'] = _import_target(arg.get('target', None), target, name='target')

    # parse rpc
    main_rpc, rpc_args = _parse_rpc(rpc, [arg.get('rpc', None) for arg in args])

    # main & sub args
    main = {'target': main}
    if main_rpc: 
        main['rpc'] = main_rpc
    for i, arg in enumerate(args):
        rpc = rpc_args[i]
        if not rpc: continue
        arg['rpc'] = rpc

    # fill shared
    _main = main
    main = shared_args.copy()
    main.update(_main)
    for i, arg in enumerate(args):
        _arg = arg
        arg = args[i] = shared_args.copy()
        arg.update(_arg)
    
    return {
        'num_process': num_process,
        'main': main,
        'args': args,
        'manager': manager,
    }
    

def _parse_rpc(shared_rpc, rpc_args):
    # check has rpc
    num_process = len(rpc_args)
    assert_type(shared_rpc, dict, null=True)
    num_sub_rpc = 0
    for rpc in rpc_args:
        assert_type(rpc, dict, null=True)
        if not rpc: continue 
        num_sub_rpc += 1
    if not shared_rpc and num_sub_rpc == 0:
        return None, rpc_args

    # main rpc
    enable_main_rpc = shared_rpc.get('enable_main_rpc', False)
    assert_type(enable_main_rpc, bool, null=False)

    # word size
    world_size = shared_rpc.get('world_size', None) if shared_rpc else None
    for rpc in rpc_args:
        if rpc is None: continue
        ws = rpc.get('world_size', None)
        if ws is None: continue
        if not world_size:
            world_size = ws
        if world_size != ws:
            raise ValueError(f'different word_size, {world_size} != {ws}')
    assert_type(world_size, int, null=False)
    if (
        (enable_main_rpc and world_size < num_sub_rpc + 1) or
        (not enable_main_rpc and world_size < num_sub_rpc)
    ): 
        raise ValueError(
            f'world_size {world_size} less than number of rpc process {num_process} with enable_main_rpc = {enable_main_rpc}'
        )

    # rank
    sub_ranks = set()
    for rpc in rpc_args:
        if rpc is None: continue
        rank = rpc.get('rank', None)
        assert_type(rank, int, null=True)
        if rank is None: continue
        if rank in sub_ranks:
            raise ValueError(f'repeated rank {rank}')
        sub_ranks.add(rank)

    if len(sub_ranks) != num_sub_rpc:
        raise ValueError(f'not all rpc has rank field, {len(sub_ranks)}/{num_sub_rpc}')

    # rank start
    rank_start = shared_rpc.get('rank', None) if shared_rpc else None
    if (
        len(sub_ranks) > 0 and
        (rank_start is not None or enable_main_rpc)
    ):
        raise ValueError(f'can not assign shared rank or enable_main_rpc with subrank at same time')

    # dispatch rank
    main_rpc = None
    if len(sub_ranks) == 0:
        if rank_start is None:
            rank_start = 0

        if enable_main_rpc:
            main_rpc = {'rank': rank_start}
            rank = rank_start + 1
        else:
            rank = rank_start

        for i, rpc in enumerate(rpc_args):
            if num_sub_rpc == 0:
                rpc = rpc_args[i] = {}
            if rpc is None: continue
            rpc['rank'] = rank
            rank += 1

    # fill shared
    shared_rpc = shared_rpc or {}
    if main_rpc:
        _main = main_rpc
        main_rpc = shared_rpc.copy()
        main_rpc.update(_main)

    for i, rpc in enumerate(rpc_args):
        if rpc is None: continue
        rpc = shared_rpc.copy()
        rpc.update(rpc_args[i])
        rpc_args[i] = rpc
    
    return main_rpc, rpc_args


def _import_target(tp, dft_tp, name):
    if tp is None:
        tp = dft_tp
    elif isinstance(tp, str):
        tp = import_type(tp)
    elif isinstance(tp, type):
        assert_subclass(tp, Target)
    elif not callable(tp):
        raise TypeError(f'invalid target type {tp} of {name}')
    return tp

def _import_manager(manager):
    assert_type(manager, str, type, null=True)
    if manager is None:
        manager = Manager
    elif isinstance(manager, str):
        manager = import_type(manager)
    assert_subclass(manager, Manager)
    return manager