import copy
import os
from collections import defaultdict
from torch import multiprocessing as mp
from torzilla.multiprocessing.target import Target, _register_target
from torzilla.threading import Result, Thread, Event
from . import parser


def launch(*args, **kwargs):
    return _launch(None, *args, **kwargs)

def launch_async(*args, **kwargs):
    launcher = Launcher(*args, **kwargs)
    Thread(target=_launch, args=[launcher] + list(args), kwargs=kwargs).start()
    return launcher._result

def _launch(launcher, *args, **kwargs):
    launcher = Launcher(*args, **kwargs) if launcher is None else launcher
    launcher.start()
    return launcher._main_target


class Launcher(object):
    def __init__(
        self,
        num_process=None,
        main=None,
        manager=None,
        target=None,
        args=None,
        rpc=None,
        **shared_args,
    ):
        # parse args
        parsed_args = parser.parse(
            num_process,
            main,
            manager,
            target,
            args,
            rpc,
            **shared_args
        )

        # processes
        self._processes = defaultdict(dict)

        # init components
        num_process = parsed_args['num_process']
        self._manager = parsed_args['manager']()
        self._manager._launcher = self
        self._barrier = mp.Barrier(num_process + 1)
        self._result = Result()
        self._start_event = Event()

        # main process
        arg = copy.copy(parsed_args['main'])
        target = arg['target']
        del arg['target']
        f_target = None
        if not isinstance(target, type) or not issubclass(target, Target):
            target, f_target = Target, target
        entry = {
            'index': None,
            'num_process': num_process,
            'manager': self._manager,
            'barrier': self._barrier,
            'target': f_target,
            'kwargs': arg,
        }
        p = mp.current_process()
        self._processes[p.pid].update({
            'process': p,
            'target': target,
            'entry': entry,
        })
        self._main_target = target(**entry)
        _register_target(self._main_target)

        # manager
        self._manager.start()
        self._manager._processes = self._manager.dict()
        
        # subprocess
        for i, arg in enumerate(parsed_args['args']):
            target = arg['target']
            del arg['target']
            f_target = None
            if not isinstance(target, type) or not issubclass(target, Target):
                target, f_target = Target, target
            entry = {
                'index': i,
                'num_process': num_process,
                'manager': self._manager,
                'barrier': self._barrier,
                'target': f_target,
                'kwargs': arg,
            }
            name = target.__name__.split('.')[-1].split('/')[-1]
            p = mp.Process(target=__subprocess_entry__, args = (target, entry), name=name)
            p.start()
            self._processes[p.pid].update({
                'process': p,
                'target': target,
                'entry': entry,
            })

        # process to manager
        for pid, info in self._processes.items():
            self._manager._processes[pid] = info['process'].name

    def start(self):
        # start
        self._main_target.start()
        self._start_event.set()
        
        # run
        self._main_target.run()

        # exit
        self._main_target.exit()
        self._manager.exit()

        # wait subprocess
        for pid, info in self._processes.items():
            if pid == os.getpid(): continue
            p = info['process']
            p.join()
        
        # set result
        self._result._set(0, (True, self._main_target))


def __subprocess_entry__(f_target, entry):
    target = f_target(**entry)
    target.start()
    target.run()
    target.exit()