from torch import multiprocessing as mp
from torzilla.multiprocessing.target import Target, _register_target
from torzilla.threading import Result, Thread, Event
from . import parser


def lanuch(*args, **kwargs):
    return _launch(*args, **kwargs)


def lanuch_async(*args, **kwargs):
    result = kwargs['result__'] = Result()
    start_event = kwargs['start_event__'] = Event()
    Thread(target=_launch, args=args, kwargs=kwargs).start()
    start_event.wait()
    return result
        

def _launch(
    num_process=None,
    main=None,
    manager=None,
    target=None,
    args=None,
    rpc=None,
    result__=None,
    start_event__=None,
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
    start_event = start_event__
    
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
    if start_event: start_event.set()

    
    # run
    main.run()

    # exit
    main.exit()
    manager.exit()

    # wait subprocess
    for p in processes:
        p.join()
    
    if result:
        result._set(0, (True, main))

    return main

def __subprocess_entry__(index, target, manager, num_process, barrier, kwargs):
    if isinstance(target, type):
        target = target(index, manager, num_process, barrier, **kwargs)
    else:
        target = Target(index, manager, num_process, barrier, target__= target, **kwargs)

    with target:
        target.run()