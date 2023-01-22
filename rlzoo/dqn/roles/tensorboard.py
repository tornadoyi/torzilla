import torch
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
        master = self.manager().writer
        self._writer = SummaryWriter(master=master)

    @staticmethod
    def guess(tag, v, global_step):
        indicators = []
        v = torch.as_tensor(v)
        if v.dtype != torch.float:
            v = v.to(torch.float32)
        if v.dim() == 1:
            indicators.append(('add_histogram', (tag + '_dist', v, global_step)))
        if v.dim() > 0:
            v = torch.mean(v)
        indicators.append(('add_scalar', (tag, v, global_step)))
        return indicators