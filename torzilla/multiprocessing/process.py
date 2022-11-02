import os
import inspect
import torch.multiprocessing as mp
from torzilla.core import utility as U, object, threading
from torzilla import rpc

__LOCK__ = mp.RLock()
__PROCESSES__ = {}
__MAIN_PID__ = None

def _add_process(process):
    global __PROCESSES__, __MAIN_PID__, __LOCK__
    with __LOCK__:
        if os.getpid() in __PROCESSES__:
            raise Exception(f'Process must be singleton, pid: {os.getpid()}')
        __PROCESSES__[os.getpid()] = process
        if isinstance(process, MainProcess):
            __MAIN_PID__ = os.getpid()

def _remove_process():
    global __PROCESSES__, __MAIN_PID__, __LOCK__
    with __LOCK__:
        process = __PROCESSES__.get(os.getpid(), None)
        if process is None: return
        del __PROCESSES__[os.getpid()]
        if os.getpid() == __MAIN_PID__:
            __MAIN_PID__ = None

def _current():
    with __LOCK__:
        return __PROCESSES__[os.getpid()]

def _main():
    with __LOCK__:
        return __PROCESSES__[__MAIN_PID__]



class Process(object.Context):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._rref = None

    @property
    def kwargs(self): return self._kwargs

    @property
    def rref(self): 
        return self._rref

    @staticmethod
    def current(): return _current()
        
    @staticmethod
    def main(): return _main()

    @staticmethod
    def current_rref(): return _current().rref
        
    def start(self): 
        _add_process(self)
        self._init_rpc()
        super().start()

    def exit(self):
        super().exit()
        if rpc.is_init():
            rpc.shutdown()
        _remove_process()

    def _init_rpc(self):
        kwargs = self.kwargs.get('rpc', None)
        if kwargs is None: return False
        keys = inspect.getfullargspec(rpc.init_rpc).args
        rpc_args = U.pick_args(kwargs, keys, drop_none=True)
        rpc.init_rpc(**rpc_args)
        self._rref = rpc.RRef(self)
        rpc.barrier()


class MainProcess(Process):
    def __init__(self, subproc_args, manager=None, **kwargs) -> None:
        super().__init__(
            subproc_args=subproc_args, 
            manager=manager,
            **kwargs
        )
        # check
        from .manager import Manager
        manager_type = manager or Manager
        U.assert_subclass(manager, Manager)
        for args in subproc_args:
            subproc = args.get('subproc', Subprocess)
            U.assert_subclass(subproc, Subprocess)

        self._subproc_args = subproc_args
        self._manager = manager_type()
        self._processes = []

    @property
    def num_process(self): return len(self._subproc_args)

    @property
    def manager(self): return self._manager

    def start(self): 
        _add_process(self)
        self._manager.start()
        self._spawn()
        _remove_process()
        super().start()
        
    def exit(self):
        super().exit()
        for p in self._processes:
            p.join()
        self._manager.exit()

    def _spawn(self):
        for i in range(len(self._subproc_args)):
            args = self._subproc_args[i]
            subproc = args['subproc']
            p = mp.Process(target=subproc._on_process_entry, args = (i, subproc, self.manager, args))
            p.start()
            self._processes.append(p)


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