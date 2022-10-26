

class BaseReplayBuffer(object):
    def __init__(self, capacity) -> None:
        self._capacity = capacity
        
    @property
    def capacity(self): return self._capacity

    def start(self):
        self._on_start()

    def _on_start(self): pass