from torzilla import multiprocessing as mp
from torzilla.core import object


class BaseBufferStore(object.Context):
    def __init__(self, capacity):
        super().__init__()
        self._capacity = capacity

    @property
    def capacity(self): self._capacity

    @property
    def running(self): return self._running.value

    def __len__(self): 
        raise NotImplementedError(f'{type(self)}.__len__ is not implemented')

    def _on_start(self):
        manager = mp.Process.current().manager
        self._running = manager.Value('b', True)

    def _on_exit(self):
        self._running.value = False
    

class BaseReplayBuffer(object.Context):
    def __init__(self, store, sampler=None):
        self._store = store
        self._sampler = sampler

    @property
    def store(self): return self._store

    @property
    def running(self): return self._running

    def __len__(self): 
        return len(self._store)

    def sample(self, *args, **kwargs):
        return self._sampler(self, *args, **kwargs)

    def sample_with(self, sampler, *args, **kwargs):
        return sampler(self, *args, **kwargs)
