

class BaseReplayBuffer(object):
    def __init__(self, max_size) -> None:
        self._max_size = max_size
        

    @property
    def max_size(self): return self._max_size