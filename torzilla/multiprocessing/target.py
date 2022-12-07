import inspect
from enum import Enum
import torch.multiprocessing as mp
from torzilla.core import pick_args
from torzilla import rpc
from torzilla import threading

def current_target():
    proc = mp.current_process()
    t = getattr(proc, '__process_target__', None)
    if t is None:
        raise RuntimeError('process target not found')
    return t

def current_target_rref():
    return current_target().rref()

def _register_target(target):
    proc = mp.current_process()
    _target = getattr(proc, '__process_target__', None)
    if _target is not None and _target != target:
        raise RuntimeError('process target register repeatedly')
    mp.current_process().__process_target__ = target

def _unregister_target():
    del mp.current_process().__process_target__


class State(Enum):
    Init = 0
    Start = 1
    Run = 2
    Exit = 3


class Target(object):
    def __init__(
        self, 
        index__,
        manager__,
        num_process__,
        barrier__,
        *,
        target__=None,
        **kwargs
    ):
        self._index = index__
        self._manager = manager__
        self._target = target__
        self._num_process = num_process__
        self._barrier = barrier__
        self._kwargs = kwargs
        self._rref = None
        self._state = State.Init
        self._rpc_evt = threading.Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc, val, tb):
        self.exit()

    def __call__(self):
        self.run()

    def index(self):
        return self._index

    def num_process(self):
        return self._num_process

    def kwargs(self): 
        return self._kwargs

    def manager(self):
        return self._manager

    def state(self):
        return self._state

    def rref(self):
        if self._rref:
            return self._rref
        if not rpc.is_init():
            raise RuntimeError(f'access RRef from rpc uninitialized target {self}')
        self._rpc_evt.wait()
        return self._rref
        
    def start(self):
        self._next_state(State.Start)
        _register_target(self)
        self._init_rpc()
        self._start()

    def run(self):
        self._next_state(State.Run)
        self._run()

    def exit(self):
        self._next_state(State.Exit)
        self._exit()
        if rpc.is_init():
            rpc.shutdown()
        _unregister_target()

    def _init_rpc(self):
        kwargs = self._kwargs.get('rpc', None)
        if kwargs is None: return False
        keys = inspect.getfullargspec(rpc.init_rpc).args
        rpc_args = pick_args(kwargs, keys, drop_none=True)
        rpc.init_rpc(**rpc_args)
        self._rref = rpc.RRef(self)
        self._rpc_evt.set()

    def _start(self): pass

    def _run(self):
        if self._target is None:
            return
        self._target(self)

    def _exit(self): pass

    def _next_state(self, state):
        if state.value <= self._state.value:
            raise RuntimeError(
                f'state transition error, {self._state.name} -> {state.name}'
            )
        self._barrier.wait()
        self._state = state