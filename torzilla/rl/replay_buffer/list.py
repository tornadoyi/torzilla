import time
from queue import Empty
from operator import itemgetter
from tkinter import E
import numpy as np
from torzilla import multiprocessing as mp
from torzilla.core import threading
from .base import BaseReplayBuffer, BaseBufferStore


class ListBufferStore(BaseBufferStore):
    def __init__(self, capacity, max_cache_size=0):
        super().__init__(capacity)
        self._max_cache_size = max_cache_size

    @property
    def max_cache_size(self): return self._max_cache_size

    @property
    def cache_size(self): return self._Q.qsize()

    def __len__(self):
        with self._L.reader_lock():
            return self._size()

    def _size(self):
        h, t = self._head.value, self._tail.value
        return self.capacity if h == t else t - h

    def _on_start(self):
        super()._on_start()
        manager = mp.Process.current().manager
        self._Q = mp.Queue(maxsize=self._max_cache_size)
        self._L = manager.RWLock()
        self._M = manager.list([None] * self.capacity)
        self._head = manager.Value('l', -1)
        self._tail = manager.Value('l', 0)
    

class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, store, enable_flush=True, cache_wait_timeout=0.5):
        super().__init__(store, self._sample)
        self._enable_flush = enable_flush
        self._cache_wait_timeout = cache_wait_timeout
        self._flush_thread = threading.Thread(target=self._flush) if enable_flush else None

    @property
    def enable_flush(self): return self._enable_flush

    @property
    def cache_wait_timeout(self): return self._cache_wait_timeout

    @property
    def cache_size(self): return self._store.cache_size
            
    def _on_start(self):
        super()._on_start()
        if self._flush_thread: 
            self._flush_thread.start()

    def _on_exit(self):
        super()._on_exit()
        if self._flush_thread:
            self._flush_thread.join()

    def put(self, data):
        self.store._Q.put(data)

    @staticmethod
    def _sample(self, batch_size):
        M, L = self.store._M, self.store._L
        with L.reader_lock():
            # check
            size = self.store._size()
            if batch_size > size:
                raise Exception(f'Not enough datas, current: {size}  expected: {batch_size}')
            
            # random
            head = self.store._head.value
            offsets = np.random.randint(low=0, high=size, size=batch_size)
            indexes = [(head + off) % self.capacity for off in offsets]
            return itemgetter(*indexes)(M)
            
    def _flush(self):
        Q, M, L = self.store._Q, self.store._M, self.store._L
        head, tail = self.store._head, self.store._tail

        def _save(datas):
            raw_data_len = len(datas)
            if raw_data_len == 0: return

            # keep num of capacity datas
            datas = datas[-self.capacity:]
            offset = raw_data_len % self.capacity if raw_data_len > self.capacity else 0

            with L.writer_lock():
                _h, _t = head.value, tail.value
                t = _t + offset
                st = 0
                if t + len(datas) >= self.capacity:
                    M[t:self.capacity] = datas[:self.capacity-t]
                    st = self.capacity - t
                    t = 0
                
                nt = t + len(datas) - st
                assert(nt == (_t + len(datas)) % self.capacity)
                M[t:nt] = datas[st:]
                
                if _h != _t and _t + raw_data_len < self.capacity:
                    head.value = 0
                else:
                    head.value = nt

                tail.value = nt 

        def _get_one(timeout):
            try:
                return Q.get(timeout=timeout)
            except Empty:
                return None
        
        cache = []
        while self.running:
            cache.clear()
            for _ in range(Q.qsize()):
                data = _get_one(self._cache_wait_timeout)
                if data is None: break
                cache.append(data)
            
            if len(cache) == 0: continue
            _save(cache)