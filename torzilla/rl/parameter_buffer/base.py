

class BaseParameterBuffer(object):
    def __init__(self, master=None, manager=None):
        super().__init__()
        self._master = master
        self._manager = manager

    def is_master(self): return self._master is None

    def __len__(self): 
        raise NotImplementedError(f'{type(self)}.__len__ is not implemented')

    def put(self):
        raise NotImplementedError(f'{type(self)}.put is not implemented')

    def get(self):
        raise NotImplementedError(f'{type(self)}.sample is not implemented')