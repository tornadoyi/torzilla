import unittest
import tempfile
import torzilla.multiprocessing as mp
from torzilla import rpc
from torch import futures

class Manager(mp.Manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _start(self):
        self.shared = self.Value('i', -1)

    def _exit(self):
        self.result = self.shared.value
    
   
def _run_test(self):
    kwargs = self.kwargs()
    run_index = kwargs['run_index']
    if self.index() != run_index: 
        return

    rrefs = futures.wait_all([
        rpc.rpc_async(info, mp.current_target_rref)
        for info in rpc.get_worker_infos()
    ])
    
    result = 0
    for rref in rrefs:
        result += rref.rpc_sync().get_id()

    self.manager().shared.value = result


class Target(mp.Target):
    def _run(self):
        _run_test(self)

    def get_id(self):
        return self.kwargs()['id']

class MockTarget():
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def rref(self):
        return rpc.RRef(self)

def _extra_target_func(id, rank, world_size, init_method):
    from torzilla.multiprocessing.target import _register_target
    mock = MockTarget(id)
    _register_target(mock)
    rpc.init_rpc(rank, world_size, init_method=init_method)
    rpc.shutdown()

    
class TestRPC(unittest.TestCase):
    num_process = 5
    
    @unittest.skip('this case effect rpc init once again')
    def test_rpc_enable_main(self):
        file = tempfile.NamedTemporaryFile()
        init_method = f'file://{file.name}'
        result = mp.launch(
            num_process=self.num_process,
            main=Target,
            target=Target,
            manager=Manager,
            rpc = {
                'enable_main_rpc': True,
                'init_method': init_method,
                'world_size': self.num_process + 1,
            },
            run_index = None,
            id = 1
        ).manager().result
        self.assertEqual(result, self.num_process+1)

    def test_rpc_rank_start(self):
        file = tempfile.NamedTemporaryFile()
        init_method = f'file://{file.name}'
        extra_procs = []
        for i in range(2):
            p = mp.Process(
                target=_extra_target_func, 
                args=(1, i, self.num_process, init_method)
            )
            p.start()
            extra_procs.append(p)

        result = mp.launch(
            num_process=self.num_process-2,
            main=Target,
            target=Target,
            manager=Manager,
            rpc = {
                'init_method': init_method,
                'world_size': self.num_process,
                'rank': 2,      # start rpc from rank=2
            },
            run_index = 2,
            id = 1
        ).manager().result

        for p in extra_procs:
            p.join()
        self.assertEqual(result, self.num_process)

    def test_rpc_diy(self):
        file = tempfile.NamedTemporaryFile()
        init_method = f'file://{file.name}'
        result = mp.launch(
            main=Target,
            target=Target,
            manager=Manager,
            args = [
                {'rpc': {'rank': 0}, 'id': 1},
                {},
                {'rpc': {'rank': 1}, 'id': 2},
                {'rpc': {'rank': 2}, 'id': 3},
                {}
            ],

            # shared rpc config
            rpc = {
                'init_method': init_method,
                'world_size': 3,
            },
            run_index = 0,
            id = 0
        ).manager().result
        self.assertEqual(result, 6)     # sum(id) = 3 + 2 + 1



if __name__ == '__main__':
    unittest.main()