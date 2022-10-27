import time
from queue import Empty
from torzilla import multiprocessing as mp
from .base import BaseReplayBuffer


class ListReplayBufferCore(object):
    pass

class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, send_qsize=0, flush_timeout=0):
        super().__init__(capacity)
        self._send_qsize = send_qsize
        self._flush_timeout = flush_timeout

    def __len__(self):
        head, tail, L = self._M['head'], self._M['tail'], self._M['L']
        with L.reader_lock():
            h, t = head.value, tail.value
        if h < 0: return 0
        return self.capacity if h == t else t - h

    def _on_start(self):
        self._M = self._get_shared_data()
        
    def push(self, data):
        self._M['Q'].put(data)

    def sample(self, batch_size):
        pass
    
    def _flush(self):
        Q, M, L = self._M['Q'], self._M['M'], self._M['L']
        head, tail = self._M['head'], self._M['tail']
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

            with L.write_lock():
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

        while True:
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
                              
            
    def _get_shared_data(self):
        manager = mp.Process.instance().manager
        with manager.shared_lock:
            if 'replay_buffer' not in manager.shared_data:
                manager.shared_data['replay_buffer'] = manager.dict(
                    Q = mp.Queue(maxsize=self._send_qsize),
                    L = manager.RWLock(),
                    M = manager.list([None] * self.capacity),
                    head = manager.Value('l', -1),
                    tail = manager.Value('l', 0),
                )
        return manager.shared_data['replay_buffer']