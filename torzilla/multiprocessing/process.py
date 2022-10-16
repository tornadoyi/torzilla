import os
import copy
import inspect
import torch.multiprocessing as mp
from torzilla.core import utility as U
from torzilla import rpc

class Process(object):
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
    
    @property
    def pid(self): return os.getpid()

    @property
    def kwargs(self): return self._kwargs

    def start(self):
        self._on_start()

    def exit(self): 
        self._on_exit()

    def _on_start(self): pass

    def _on_exit(self): pass

    def _init_rpc(self, **kwargs):
        keys = inspect.getfullargspec(rpc.init_rpc).args
        rpc_args = U.pick_args(kwargs, keys, drop_none=True)
        return rpc.init_rpc(**rpc_args)



class MainProcess(Process):
    def _on_start(self):
        num_process = self.kwargs.get('num_process', 0)
        rpc_kwargs = self.kwargs.get('rpc', None)
        U.assert_type(num_process, int)

        # start rpc
        if num_process == 0 and rpc_kwargs is not None:
            self._init_rpc(rpc_kwargs)
            rpc.shutdown()

        # start subprocess
        else:
            ex_kwargs = self._pre_spawn()
            U.assert_type(ex_kwargs, dict, null=True, name='pre_spwan_kwargs')
            spw_kwargs = self.kwargs
            if ex_kwargs is not None:
                spw_kwargs = copy.deepcopy(spw_kwargs) 
                spw_kwargs.update(ex_kwargs)
            self._spawn(num_process, spw_kwargs)
    
    def _pre_spawn(self): return None
                
    def _spawn(self, num_process, kwargs):
        # check
        subproc = kwargs.get('subproc', Subprocess)
        if isinstance(subproc, str): subproc = U.import_type(subproc)
        U.assert_subclass(subproc, Subprocess)

        # create manager
        manager = mp.Manager()

        # start
        processes = []
        for i in range(num_process):
            p = mp.Process(target=subproc._on_process_entry, args = (i, subproc, manager, kwargs))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    
class Subprocess(Process):
    def __init__(self, manager, **kwargs) -> None:
        super().__init__(**kwargs)
        self._manager = manager
        self._proc_index = kwargs.get('proc_index', None)
        U.assert_type(self._proc_index, int, name='proc_index')
    
    @property
    def proc_index(self): return self._proc_index

    @property
    def manager(self): return self._manager

    @staticmethod
    def _on_process_entry(index, proc, manager, kwargs):
        try:
            p = None
            p = proc(proc_index=index, manager=manager, **kwargs)
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
