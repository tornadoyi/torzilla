import torch
import numpy as np
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
        
        # writer
        self._writer = SummaryWriter(**cfg_tb)

    def add_ops(self, ops):
        for item in ops:
            if len(item) == 2:
                fname, args = item
                kwds = {}
            elif len(item) == 3:
                fname, args, kwds = item
            else:
                raise ValueError(f'length of op is {len(item)}, expected: 2,3')
            f = getattr(self, fname)
            f(*args, **kwds)

    @staticmethod
    def make_numeric_ops(tag, v, global_step):
        ops = []
        v = torch.as_tensor(v)
        if v.dtype != torch.float:
            v = v.to(torch.float32)
        if v.dim() == 1:
            ops.append(('add_histogram', (tag + '_dist', v, global_step)))
        if v.dim() > 0:
            v = torch.mean(v)
        ops.append(('add_scalar', (tag, v, global_step)))
        return ops

