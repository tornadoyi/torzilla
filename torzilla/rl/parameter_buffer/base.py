from torzilla.core import object

class BaseParameterBuffer(object.Context):
    def __init__(self, master=None):
        super().__init__()
        self._master = master

    @property
    def master(self): return self._master

    def is_master(self): return self._master is None

    def __len__(self): 
        raise NotImplementedError(f'{type(self)}.__len__ is not implemented')

    def put(self, *args, **kwargs):
        raise NotImplementedError(f'{type(self)}.put is not implemented')

    def get(self, *args, **kwargs):
        raise NotImplementedError(f'{type(self)}.sample is not implemented')