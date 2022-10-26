import os
import copy
import inspect
import torch.multiprocessing as mp
from torzilla.core import utility as U
from torzilla import rpc

__PROCESSES__ = {}

class Process(object):
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    @property
    def kwargs(self): return self._kwargs

    @staticmethod
    def instance(): 
        return __PROCESSES__.get(os.getpid(), None)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc, val, tb):
        self.exit()

    def start(self): 
        global __PROCESSES__
        if os.getpid() in __PROCESSES__:
            raise Exception(f'Process must be singleton, pid: {os.getpid()}')
        __PROCESSES__[os.getpid()] = self
        self._on_start()

    def exit(self):
        global __PROCESSES__
        self._on_exit()
        if os.getpid() not in __PROCESSES__: return
        del __PROCESSES__[os.getpid()]

    def _on_start(self): pass

    def _on_exit(self): pass

    @staticmethod
    def _init_rpc(**kwargs):
        keys = inspect.getfullargspec(rpc.init_rpc).args
        rpc_args = U.pick_args(kwargs, keys, drop_none=True)
        return rpc.init_rpc(**rpc_args)


class MainProcess(Process):
    def __init__(self, subproc_args, manager=None, **kwargs) -> None:
        super().__init__(
            subproc_args=subproc_args, 
            manager=manager,
            **kwargs
        )
        # check
        from .manager import Manager
        manager = manager or Manager
        U.assert_subclass(manager, Manager)
        for args in subproc_args:
            subproc = args.get('subproc', Subprocess)
            U.assert_subclass(subproc, Subprocess)

        self._manager_type = manager
        self._subproc_args = subproc_args
        self._manager = None

    @property
    def num_process(self): return len(self._subproc_args)

    @property
    def manager(self): return self._manager

    def _on_start(self):
        with self._create_manager():
            self._spawn()

    def _create_manager(self):
        self._manager = self._manager_type()
        return self._manager

    def _spawn(self):
        processes = []
        for i in range(len(self._subproc_args)):
            args = self._subproc_args[i]
            subproc = args['subproc']
            p = mp.Process(target=subproc._on_process_entry, args = (i, subproc, self.manager, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


class Subprocess(Process):
    def __init__(self, index, manager, **kwargs) -> None:
        super().__init__(**kwargs)
        self._manager = manager
        self._index = index
        U.assert_type(self._index, int, name='index')
    
    @property
    def index(self): return self._index

    @property
    def manager(self): return self._manager

    @staticmethod
    def _on_process_entry(index, proc, manager, kwargs):
        with proc(index=index, manager=manager, **kwargs):
            pass

    def _on_start(self):
        if self._try_init_rpc():
            rpc.shutdown()

    def _try_init_rpc(self):
        rpc_kwargs = self.kwargs.get('rpc', None)
        if rpc_kwargs is None: return False

        # check
        rank_start = rpc_kwargs.get('rank', None)
        num_rpc = rpc_kwargs.get('num_rpc', float('inf'))
        U.assert_type(rank_start, int)
        U.assert_type(num_rpc, int, float)
        if self.index >= num_rpc: return False

        # start
        rpc_kwargs = copy.copy(rpc_kwargs)
        rpc_kwargs['rank'] = self.index + rank_start
        self._init_rpc(**rpc_kwargs)

        return True
