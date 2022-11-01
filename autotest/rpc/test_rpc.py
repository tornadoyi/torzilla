import unittest
import random
import tempfile
import torzilla.multiprocessing as mp
from torzilla import rpc

class Manager(mp.Manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _on_start(self, *args, **kwargs):
        self._shared = self.Value('i', 1)

    def _on_exit(self, *args, **kwargs):
        self._result = self._shared.value
    
    def get_shared(self): return self._shared

    def get_result(self): return self._result


def run(self):
    world_size = self.kwargs['rpc']['world_size']
    futures = []
    for i in range(world_size):
        fut = rpc.rpc_async(str(i), mp.Process.current_rref)
        futures.append(fut)
    
    rrefs = [fut.wait() for fut in futures]
    
    result = 0
    for i, rref in enumerate(rrefs):
        rank = rref.rpc_sync().get_rank()
        result += i - rank

    self.manager.get_shared().value = result


class MainProcess(mp.MainProcess):
    def _on_start(self):
        if self.kwargs['run_index'] == 'main':
            run(self)
    
    def get_rank(self):
        return rpc.get_worker_info().id


class Subprocess(mp.Subprocess):
    def _on_start(self):
        if self.kwargs['run_index'] == self.index:
            run(self)

    def get_rank(self):
        return rpc.get_worker_info().id

    
class TestProcess(unittest.TestCase):
    
    @unittest.skip('this case effect rpc init once again')
    def test_rpc_mainproc(self):
        file = tempfile.NamedTemporaryFile()
        num_process = 5
        mp.lanuch(
            num_process=num_process,
            mainproc=MainProcess,
            subproc=Subprocess,
            manager=Manager,
            rpc = {
                'enable_mainproc_rpc': True,
                'init_method': f'file://{file.name}'
            },
            run_index = 'main'
        )

    def test_rpc_subproc(self):
        num_process = 5
        file = tempfile.NamedTemporaryFile()
        proc = mp.lanuch(
            num_process=num_process,
            mainproc=MainProcess,
            subproc=Subprocess,
            manager=Manager,
            rpc = {
                'init_method': f'file://{file.name}'
            },
            run_index = random.randint(0, num_process-1)
        )
        self.assertEqual(proc.manager.get_result(), 0)

    def test_rpc_subproc_diy(self):
        num_process = 5
        file = tempfile.NamedTemporaryFile()
        subargs = []
        for i in range(num_process):
            args = {
                'rpc': {
                    'init_method': f'file://{file.name}',
                    'rank': i,
                    'world_size': num_process
                }
            }
            subargs.append(args)

        proc = mp.lanuch(
            num_process=num_process,
            mainproc=MainProcess,
            subproc=Subprocess,
            manager=Manager,
            subproc_args=subargs,
            run_index = random.randint(0, num_process-1)
        )
        self.assertEqual(proc.manager.get_result(), 0)


if __name__ == '__main__':
    unittest.main()