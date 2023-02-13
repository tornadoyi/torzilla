import unittest
import shutil
from torch import futures
import numpy as np
import tempfile
import random
import torch
import torzilla.multiprocessing as mp
from torzilla.rl.tensorboard import SummaryWriter
from torzilla import rpc



class Tensorboard(mp.Target):
    def _start(self):
        self._writer = SummaryWriter()
        
    def add_scalar(self, *args, **kwargs):
        return self._writer.add_scalar(*args, **kwargs)


class TestTarget(mp.Target):
    def _start(self):
        self.tb = rpc.rpc_sync('0', mp.current_target_rref)
    
    def _run(self):
        SHAPE = (10, 10)
        for i in range(10):
            t = torch.mean(torch.rand(*SHAPE))
            self.tb.rpc_sync().add_scalar('test', t, global_step=i)


class TestWriter(unittest.TestCase):
    
    def setUp(self) -> None:
        shutil.rmtree('./runs', ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree('./runs', ignore_errors=True)

    def test_tb_writer(self):
        file = tempfile.NamedTemporaryFile()
        args = [{'target': Tensorboard}] + [{'target': TestTarget}] * random.randint(3, 5)
        mp.lanuch(
            args = args,
            rpc = {
                'init_method': f'file://{file.name}',
                'world_size': len(args)
            },
        )



if __name__ == '__main__':
    unittest.main()