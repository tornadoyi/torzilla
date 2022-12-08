import torzilla.multiprocessing as mp

class BaseParameterBuffer(object):
    def __init__(self, master=None, manager=None):
        super().__init__()
        self._master = master
        self._manager = manager or mp.current_target().manager()

    def is_master(self): return self._master is None

    def manager_create(self, name, *args, **kwargs):
        f_create = getattr(self._manager, name, None)
        if f_create is None:
            raise AttributeError(f'{name} depended by {self} is not found in manager, need register first')
        return f_create(*args, **kwargs)

    def __len__(self): 
        raise NotImplementedError(f'{type(self)}.__len__ is not implemented')

    def put(self):
        raise NotImplementedError(f'{type(self)}.put is not implemented')

    def get(self):
        raise NotImplementedError(f'{type(self)}.sample is not implemented')