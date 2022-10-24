import os
import copy
import inspect
import torch.multiprocessing as mp
from torzilla.core import utility as U
from torzilla import rpc

__PROCESS__ = None

class Process(object):
    def __init__(self, **kwargs) -> None:
        global __PROCESS__
        __PROCESS__ = self
        self._kwargs = kwargs
    
    @property
    def pid(self): return os.getpid()

    @property
    def kwargs(self): return self._kwargs

    @staticmethod
    def instance(): return __PROCESS__

    def start(self):
        self._on_start()

    def exit(self): 
        self._on_exit()

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
        self._create_manager()
        self._spawn()


    def _create_manager(self):
        self._manager = self._manager_type()
        self._manager.start()


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
        U.assert_type(self._index, int, name='proc_index')
    
    @property
    def index(self): return self._index

    @property
    def manager(self): return self._manager

    @staticmethod
    def _on_process_entry(index, proc, manager, kwargs):
        try:
            p = None
            p = proc(index=index, manager=manager, **kwargs)
            p.start()
        finally:
            if p is not None: p.exit()

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
        if self.proc_index >= num_rpc: return False

        # start
        rpc_kwargs = copy.copy(rpc_kwargs)
        rpc_kwargs['rank'] = self.proc_index + rank_start
        self._init_rpc(**rpc_kwargs)

        return True
