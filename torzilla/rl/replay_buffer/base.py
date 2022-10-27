

class BaseReplayBufferStore(object):
    def __init__(self, capacity):
        self._capacity = capacity

    @property
    def capacity(self): self._capacity

    def __len__(self): 
        raise NotImplementedError(f'{type(self)}.__len__ is not implemented')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc, val, tb):
        self.exit()

    def start(self):
        self._on_start()

    def exit(self):
        self._on_exit()

    


class BaseReplayBuffer(object):
    def __init__(self, store):
        self._store = store
        
    @property
    def capacity(self): return self._store.capacity

    def start(self):
        self._on_start()

    def _on_start(self): pass