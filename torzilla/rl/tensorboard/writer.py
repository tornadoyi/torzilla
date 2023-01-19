import inspect
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
from torzilla import multiprocessing as mp




def __export_writer():
    dic = {}
    methods = [k for k in _SummaryWriter.__dict__.keys() if not k.startswith('_')]
    for meth in methods:
        exec(f'''
def {meth}(self, /, *args, **kwds):
    with self._lock:
        return self._writer.{meth}(*args, **kwds)'''
, dic)
    return type('__SummaryWriter__', (), dic)


class SummaryWriter(__export_writer()):
    def __init__(self, *args, manager=None, **kwargs):
        self._writer = _SummaryWriter(*args, **kwargs)
        self._manager = manager or mp.current_target().manager()
        self._lock = self._manager.Lock()
