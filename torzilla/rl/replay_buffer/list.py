import time
from queue import Empty
from operator import itemgetter
import numpy as np
from torzilla import multiprocessing as mp
from torzilla.core import threading
from .base import BaseReplayBuffer, BaseBufferStore


class ListBufferStore(BaseBufferStore):
    def __init__(self, capacity, send_qsize=0):
        super().__init__(capacity)
        self._send_qsize = send_qsize

    @property
    def send_queue(self): return self._Q

    @property
    def lock(self): return self._L

    @property
    def memory(self): return self._M

    @property
    def head(self): return self._head

    @property
    def tail(self): return self._tail

    def __len__(self):
        with self._L.reader_lock():
            return self._size()

    def _size(self):
        h, t = self._head.value, self._tail.value
        if h < 0: return 0
        return self.capacity if h == t else t - h

    def _on_start(self):
        super()._on_start()
        manager = mp.Process.current().manager
        self._Q = mp.Queue(maxsize=self._send_qsize)
        self._L = manager.RWLock()
        self._M = manager.list([None] * self.capacity)
        self._head = manager.Value('l', -1)
        self._tail = manager.Value('l', 0)
    

class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, store, enable_flush=False, flush_timeout=0):
        super().__init__(store, self._sample)
        self._enable_flush = enable_flush
        self._flush_timeout = flush_timeout
        self._flush_thread = threading.Thread(target=self._flush) if enable_flush else None

    @property
    def enable_flush(self): return self._enable_flush

    @property
    def flush_timeout(self): return self._flush_timeout

    def _on_start(self):
        super()._on_start()
        if self._flush_thread: 
            self._flush_thread.start()

    def _on_exit(self):
        super()._on_exit()
        if self._flush_thread:
            self._flush_thread.join()

    def push(self, data):
        self.store.send_queue.put(data)

    @staticmethod
    def _sample(self, batch_size):
        M, L = self.store.memory, self.store.lock
        with L.reader_lock():
            # check
            size = self.store._size()
            if batch_size > size:
                raise Exception(f'Not enough datas, current: {size}  expected: {batch_size}')
            
            # random
            head = self.store.head.value
            offsets = np.random.randint(low=0, high=size, size=batch_size)
            indexes = [(head + off) % self.capacity for off in offsets]
            return itemgetter(*indexes)(M)
            
    def _flush(self):
        Q, M, L = self.store.send_queue, self.store.memory, self.store.lock
        head, tail = self.store.head, self.store.tail
        cache = []

        def _get_one(timeout):
            try:
                return Q.get(timeout=timeout)
            except Empty:
                return None
        
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
                if t + len(datas) > self.capacity:
                    M[t:self.capacity] = datas[:self.capacity-t]
                    t = 0
                    st = self.capacity - t
                
                nt = t + len(datas) - st
                assert(nt == (_t + len(datas)) % self.capacity)
                M[t:nt] = datas[st:]
                
                if _h < 0 and _t + raw_data_len < self.capacity:
                    head.value = 0
                else:
                    head.value = nt

                tail.value = nt        

        while self.store.running:
            cache.clear()
            end_time = timeout = None
            while timeout is None or timeout > 0:
                data = _get_one(timeout)
                if data is not None: 
                    cache.append(data)
                    end_time = end_time or time.time() + self._flush_timeout
                        
                timeout = end_time if end_time is None else end_time - time.time()

                if not L.wlocked() or timeout <= 0:
                    _save(cache)
                    timeout = -1