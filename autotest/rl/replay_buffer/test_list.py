import unittest
from torch import futures
import numpy as np
import tempfile
import random
import torzilla.multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer
from torzilla import rpc


class Manager(mp.Manager):
    def _start(self):
        capacity = mp.current_target().kwargs()['capacity']
        self.buffer = ListReplayBuffer(capacity=capacity)
        self.Q_send = self.Queue()
        self.Q_recv = self.Queue()

def CreateTarget(name, export_cls):
    dic = {}
    methods = [k for k in export_cls.__dict__.keys() if not k.startswith('_')]
    for meth in methods:
        exec('''def %s(self, /, *args, **kwds):
        return self._core.%s(*args, **kwds)''' % (meth, meth), dic)
    return type(name, (mp.Target,), dic)

class ReplayBuffer(CreateTarget('_ReplayBuffer', ListReplayBuffer)):
    def _start(self):
        self._core = ListReplayBuffer(master=self.manager().buffer)
    
    def size(self):
        return len(self._core)

def _indexes_sampler(self, size, indexes):
    '''
    This is must be global function or stuck at tensorpipe, I dont know why
    '''
    # uniform sample
    return self._store[indexes]

class TestTarget(mp.Target):
    def _start(self):
        # get all replays
        rank = rpc.get_worker_info().id
        self.replay_rrefs = futures.wait_all([
            rpc.rpc_async(info, mp.current_target_rref) 
            for info in rpc.get_worker_infos() 
            if info.id != rank
        ])
        self.capacity = self.rb().rpc_sync().capacity()
        self.result = None
    
    def _run(self):
        f = getattr(self, self.kwargs()['case'])
        try:
            f()
        except Exception as e:
            if isinstance(e, InterruptedError):
                return
            self.result = e

    def _exit(self):
        self.manager().Q_send.put(self.result)

    def test_rb_list_append(self):
        num_data = random.randint(self.capacity // 4, self.capacity // 2)
        futures.wait_all([self.rb().rpc_async().append(i) for i in range(num_data)])
        self.assertEqual(self.rb().rpc_sync().size(), num_data, 'append')

        # put overflow
        futures.wait_all([self.rb().rpc_async().append(i) for i in range(self.capacity)])
        self.assertEqual(self.rb().rpc_sync().size(), self.capacity, 'append overflow')

    def test_rb_list_clear(self):
        self.init_replay_buffer()
        self.rb().rpc_sync().clear()
        self.assertEqual(self.rb().rpc_sync().size(), 0, 'clear')

    def test_rb_list_extend(self):
        num_data = random.randint(self.capacity // 4, self.capacity // 2)
        datas = [i for i in range(num_data)]
        self.rb().rpc_sync().extend(datas)
        self.assertEqual(self.rb().rpc_sync().size(), num_data, 'extend')

        self.rb().rpc_sync().extend([i for i in range(self.capacity)])
        self.assertEqual(self.rb().rpc_sync().size(), self.capacity, 'extend overflow')

    def test_rb_list_pop(self):
        datas = self.init_replay_buffer()
        
        for i in range(1, len(datas) // 2):
            self.rb().rpc_sync().pop()
            self.assertEqual(self.rb().rpc_sync().size(), len(datas)-i, 'pop')

    def test_rb_list_popn(self):
        datas = self.init_replay_buffer()
        num_pop = len(datas)  // 2
        self.rb().rpc_sync().popn(num_pop)
        self.assertEqual(self.rb().rpc_sync().size(), len(datas)-num_pop, 'popn')

    def test_rb_list_sample(self):
        datas = self.init_replay_buffer()

        self.assertEqual(
            len(self.rb().rpc_sync().sample(len(datas) // 2)), 
            len(datas) // 2, 
            'sample half'
        )

        self.assertEqual(
            len(self.rb().rpc_sync().sample(len(datas))), 
            len(datas), 
            'sample full'
        )

    def test_rb_list_sample_by(self):
        datas = self.init_replay_buffer()
        indexes = np.random.randint(low=0, high=len(datas), size=random.randint(10, 100)).tolist()
        sample_datas = self.rb().rpc_sync().sample(len(indexes), by=_indexes_sampler, indexes=indexes)
        gt_datas = [datas[i] for i in indexes]
        self.assertEqual(len(sample_datas), len(gt_datas), 'sample by length')

    def init_replay_buffer(self):
        num_data = random.randint(self.capacity // 4, self.capacity)
        datas = [i for i in range(num_data)]
        self.rb().rpc_sync().extend(datas)
        self.assertEqual(self.rb().rpc_sync().size(), num_data, 'init replay buffer')
        return datas

    def assertEqual(self, first, second, msg=None):
        Q_send = mp.current_target().manager().Q_send
        Q_recv = mp.current_target().manager().Q_recv
        Q_send.put(('assertEqual', (first, second, msg)))
        if not Q_recv.get():
            raise InterruptedError()

    def rb(self):
        idx = random.randint(0, len(self.replay_rrefs)-1)
        return self.replay_rrefs[idx]


class TestListReplayBuffer(unittest.TestCase):
    
    def setUp(self):
        file = tempfile.NamedTemporaryFile()
        args = [{'target': TestTarget}] + [{'target': ReplayBuffer}] * 1 #random.randint(1, 10)
        self.result = mp.lanuch_async(
            manager=Manager,
            args = args,
            rpc = {
                'init_method': f'file://{file.name}',
                'world_size': len(args)
            },
            capacity=100,
            case = self._testMethodName
        )

    def tearDown(self) -> None:
        self.result.get()

    def run_test(self):
        Q_send = mp.current_target().manager().Q_send
        Q_recv = mp.current_target().manager().Q_recv
        while True:
            o = Q_send.get()
            if o is None: break
            if isinstance(o, Exception):
                raise o

            fname, args = o
            try:
                getattr(self, fname)(*args)
                Q_recv.put(True)
            except Exception as e:
                Q_recv.put(False)
                raise e

    def test_rb_list_append(self):
        self.run_test()

    def test_rb_list_clear(self):
        self.run_test()

    def test_rb_list_extend(self):
        self.run_test()

    def test_rb_list_pop(self):
        self.run_test()

    def test_rb_list_popn(self):
        self.run_test()

    def test_rb_list_sample(self):
        self.run_test()

    def test_rb_list_sample_by(self):
        self.run_test()
        
        



if __name__ == '__main__':
    unittest.main()