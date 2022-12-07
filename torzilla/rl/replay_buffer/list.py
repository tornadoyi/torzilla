import numpy as np
from torzilla.multiprocessing import MakeProxyType, delegate, Manager
from torzilla.collections import clist
from .base import BaseReplayBuffer


class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity=None, **kwargs):
        super().__init__(**kwargs)
        if self.is_master():
            self._store = self.manager_create(_Store.Name, capacity)
            self._qlock = self.manager_create('RWLock')
            self._mlock = self.manager_create('RWLock')
        else:
            self._store = self._master._store

    def capacity(self): return self._capacity

    def __len__(self):
        with self._mlock.rlock(), self._qlock.rlock():
            return len(self._store)

    def append(self, value):
        with self._qlock.wlock():
            self._store.append(value)

    def extend(self, values):
        with self._qlock.wlock():
            self._store.extend(values)

    def sample(self, size, by=None): 
        def _sample(size):
            # uniform sample
            indexes = np.random.randint(low=0, high=len(self._store), size=size).tolist()
            return self._store[indexes]
        
        sampler = by or _sample
        if self._store.qsize() > 0:
            with self._qlock.wlock(), self._mlock.wlock():
                self._store.flush()
                return sampler(size)
        else:
            with self._mlock.rlock():
                return sampler(size)
           
    def clear(self):
        with self._qlock.wlock(), self._mlock.wlock():
            self._store.clear()

    def pop(self):
        with self._qlock.wlock(), self._mlock.wlock():
            self._store.pop()

    def popn(self, n=1):
        with self._qlock.wlock(), self._mlock.wlock():
            self._store.popn(n)


class _Store(object):
    Name = '_ListReplayBufferStore'

    def __init__(self, capacity):
        self._M = clist(capacity)
        self._Q = []

    def __getitem__(self, key):
        return self._Q[key]

    def __len__(self):
        return max(len(self._M) + len(self._Q), self._M.capacity())
    
    def append(self, value):
        self._Q.append(value)

    def clear(self):
        self._Q.clear()
        self._M.clear()

    def extend(self, values):
        self._Q.extend(values)

    def flush(self):
        self._M.extend(self._Q)
        self._Q.clear()

    def pop(self):
        self._M.pop()

    def popn(self, n):
        self._M.popn(n)
    
    def qsize(self):
        return len(self.Q)



class _StoreProxy (MakeProxyType('GearProxy', (
    '__getitem__', '__len__', 
    'append', 'clear', 'extend', 'flush', 'pop', 'popn', 'qsize',
))):
    pass

Manager.register(_Store.Name, delegate(_Store), _StoreProxy)