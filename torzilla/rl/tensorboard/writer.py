import inspect
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
from torzilla import multiprocessing as mp
from torzilla.multiprocessing import MakeProxyType, delegate, Manager

__WRITER_EXPOSED_METHODS__ = tuple([k for k in _SummaryWriter.__dict__.keys() if not k.startswith('_')])

def __export_writer():
    dic = {}
    for meth in __WRITER_EXPOSED_METHODS__:
        exec(f'''
def {meth}(self, /, *args, **kwds):
    with self._lock:
        return self._writer.{meth}(*args, **kwds)
'''
, dic)
    return type('__SummaryWriter__', (), dic)


class SummaryWriter(__export_writer()):
    def __init__(self, *args, master=None, manager=None, **kwargs):
        self._master = master
        self._manager = manager or mp.current_target().manager()

        if self._master is None:
            self._writer = self.manager_create(__WRITER_NAME__, *args, **kwargs)
            self._lock = self._manager.Lock()
        else:
            self._writer = self._master._writer
            self._lock = self._master._lock

    def add(self, fname, *args, **kwargs):
        f = getattr(self._writer, fname)
        with self._lock:
            return f(*args, **kwargs)

    def adds(self, indicators):
        # check
        ops = []
        for item in indicators:
            if len(item) == 2:
                fname, args = item
                kwds = {}
            elif len(item) == 3:
                fname, args, kwds = item
            else:
                raise ValueError(f'length of indicators is {len(item)}, expected: 2,3')
            ops.append((getattr(self._writer, fname), args, kwds))
        
        # call
        with self._lock:
            for (f, args, kwds) in ops:
                f(*args, **kwds)

    def manager_create(self, name, *args, **kwargs):
        f_create = getattr(self._manager, name, None)
        if f_create is None:
            raise AttributeError(f'{name} depended by {self} is not found in manager, need register first')
        return f_create(*args, **kwargs)


__WRITER_NAME__ = '__SummaryWriter__'

class _WriterProxy (MakeProxyType('__Writer', __WRITER_EXPOSED_METHODS__)):
    pass

Manager.register(__WRITER_NAME__, delegate(_SummaryWriter), _WriterProxy)
