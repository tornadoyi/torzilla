from torch import multiprocessing as mp
from torzilla.multiprocessing.target import Target, _register_target
from torzilla.threading import Result, Thread
from . import parser


def lanuch(*args, **kwargs):
    return _launch(*args, **kwargs)


def lanuch_async(*args, **kwargs):
    result = kwargs['result__'] = Result()
    Thread(target=_launch, args=args, kwargs=kwargs).start()
    return result
        

def _launch(
    result__=None,
    num_process=None,
    main=None,
    manager=None,
    target=None,
    args=None,
    rpc=None,
    **shared_args,
):
    # parse
    parsed_args = parser.parse(
        num_process,
        main,
        manager,
        target,
        args,
        rpc,
        **shared_args
    )
    num_process = parsed_args['num_process']
    main = parsed_args['main']
    args = parsed_args['args']
    manager = parsed_args['manager']
    result = result__
    
    # create barrier for sync processes
    barrier = mp.Barrier(num_process + 1)

    # init manager
    manager = manager()

    # init main
    target = main['target']
    del main['target']
    if isinstance(target, type):
        main = target(None, manager, num_process, barrier, **main)
    else:
        main = Target(None, manager, num_process, barrier, target__= target, **main)

    # init subprocess
    processes = []
    for i, arg in enumerate(args):
        target = arg['target']
        del arg['target']
        p = mp.Process(
            target=__subprocess_entry__, 
            args = (i, target, manager, num_process, barrier, arg)
        )
        processes.append(p)

    # start
    _register_target(main)
    manager.start()
    for p in processes:
        p.start()
    main.start()
    
    # run
    main.run()

    # exit
    main.exit()
    manager.exit()

    # wait subprocess
    for p in processes:
        p.join()
    
    if result:
        result._set((True, main))

    return main

def __subprocess_entry__(index, target, manager, num_process, barrier, kwargs):
    if isinstance(target, type):
        target = target(index, manager, num_process, barrier, **kwargs)
    else:
        target = Target(index, manager, num_process, barrier, target__= target, **kwargs)

    with target:
        target.run()