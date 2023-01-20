import shutil
from torzilla.rl.tensorboard import SummaryWriter
from rlzoo.zoo.role import Role

def __export_board():
    dic = {}
    methods = [k for k in dir(SummaryWriter) if not k.startswith('_')]
    for meth in methods:
        exec(f'''
def {meth}(self, /, *args, **kwds):
    return self._writer.{meth}(*args, **kwds)'''
, dic)
    return type('__Board__', (Role,), dic)


class Tensorboard(__export_board()):
    def _start(self):
        # pick writer agrs
        cfg_tb = self.kwargs()['config']['tb']
        
        # clean old log dir
        log_dir = cfg_tb.get('log_dir', None)
        if log_dir is not None:
            shutil.rmtree(log_dir, ignore_errors=True)

        # writer
        self._writer = SummaryWriter(**cfg_tb)

    def add_all(self, ops):
        for (fname, args, kwds) in ops:
            f = getattr(self, fname)
            f(*args, **kwds)

