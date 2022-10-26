from .base import BaseReplayBuffer
from torzilla import multiprocessing as mp


class ListReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)

    def _on_start(self):
        self._M = self._get_shared_data()
        

    def push(self, data):
        self._M['Q'].put(data)

    def sample(self):
        pass
    
    
    def _flush(self):
        Q, M, L = self._M['Q'], self._M['M'], self._M['L']
        head, tail = self._M['head'], self._M['tail']
        while True:
            data = Q.get()
            with L:
                M[tail.value] = data
                tail.value = (tail.value + 1) % self.capacity
                if tail.value == head.value:
                    head.value = (head.value + 1) % self.capacity

            
    def _get_shared_data(self):
        manager = mp.Process.instance().manager
        with manager.shared_lock:
            if 'replay_buffer' not in manager.shared_data:
                manager.shared_data['replay_buffer'] = manager.dict(
                    Q = mp.SimpleQueue(),
                    L = mp.RLock(),
                    M = manager.list([None] * self.capacity),
                    head = manager.Value('i', 0),
                    tail = manager.Value('i', 0),
                )
        return manager.shared_data['replay_buffer']