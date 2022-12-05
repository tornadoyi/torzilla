import numpy as np
from torzilla import multiprocessing as mp
from .base import BaseReplayBuffer


class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity=None, max_cache_size=0, **kwargs):
        super().__init__(**kwargs)
        if self.is_master():
            self._capacity = capacity
            self._max_cache_size = max_cache_size
        else:
            self._capacity = self.master.capacity
            self._max_cache_size = self.master.max_cache_size

    @property
    def capacity(self): return self._capacity

    @property
    def max_cache_size(self): return self._max_cache_size

    @property
    def cache_size(self):
        with self._qlock:
            return len(self._Q)

    @property
    def _size(self):
        return min(len(self._M) + len(self._Q), self._capacity)

    def __len__(self):
        with self._mlock.rlock(), self._qlock:
            return self._size

    def size(self):
        return len(self)

    def _on_start(self):
        super()._on_start()
        if self.is_master():
            manager = mp.Process.current().manager
            self._Q = manager.clist(self._max_cache_size)
            self._M = manager.clist(self._capacity)
            self._qlock = manager.Lock()
            self._mlock = manager.RWLock()
        else:
            self._Q = self.master._Q
            self._M = self.master._M
            self._qlock = self.master._qlock
            self._mlock = self.master._mlock

    def put(self, *datas):
        with self._qlock:
            if len(self._Q) + len(datas) > self.max_cache_size:
                with self._mlock.wlock():
                    self._flush(datas)
            else:
                self._Q.extend(datas)

    def sample(self, size): 
        with self._qlock:
            if len(self._Q) > 0:
                with self._mlock.wlock():
                    self._flush()

        with self._mlock.rlock():
            # uniform sample
            buffer_size = self._size
            indexes = np.random.randint(low=0, high=buffer_size, size=size).tolist()
            return self._M[indexes]

    def clear(self):
        with self._qlock, self._mlock.wlock():
            self._Q.clear()
            self._M.clear()

    def pop(self):
        with self._qlock:
            if len(self._Q) > 0:
                with self._mlock.wlock():
                    self._flush()

        with self._mlock.wlock():
            self._M.pop()

    def popleft(self):
        with self._qlock:
            if len(self._Q) > 0:
                with self._mlock.wlock():
                    self._flush()

        with self._mlock.wlock():
            self._M.popleft()

    def popn(self, n=1):
        with self._qlock:
            if len(self._Q) > 0:
                with self._mlock.wlock():
                    self._flush()

        with self._mlock.wlock():
            self._M.popn(n)

    def popnleft(self, n=1):
        with self._qlock:
            if len(self._Q) > 0:
                with self._mlock.wlock():
                    self._flush()

        with self._mlock.wlock():
            self._M.popnleft(n)
    
    def _flush(self, datas=None):
        if datas:
            self._M.extend(self._Q[:] + list(datas))
        else:
            self._M.extend(self._Q)
        self._Q.clear()