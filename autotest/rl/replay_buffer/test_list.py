import unittest
from torch import futures
import tempfile
import random
import torzilla.multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer
from torzilla import rpc

class Manager(mp.Manager):
    def _start(self):
        capacity = mp.current_target().kwargs['capacity']
        self.cases = self.list()
        self.buffer = ListReplayBuffer(capacity=capacity)

def CreateTarget(name, export_cls):
    dic = {}
    methods = [k for k in export_cls.__dict__.keys() if not k.startswith('_')]
    for meth in methods:
        exec('''def %s(self, /, *args, **kwds):
        return self._core(%r, *args, **kwds)''' % (meth, meth), dic)
    return type(name, (mp.Target,), dic)

class ReplayBuffer(CreateTarget('_ReplayBuffer', ListReplayBuffer)):
    def _start(self):
        self._core = ListReplayBuffer(master=self.manager().buffer)

    def size(self):
        return len(self._core)


class TestTarget(mp.Target):
    def _start(self):
        # get all replays
        rank = rpc.get_worker_info().id
        replay_ranks = [info.id for info in rpc.get_worker_infos() if info.id != rank]
        self.replay_rrefs = futures.wait_all([rpc.rpc_async(str(r), mp.Process.current_rref) for r in replay_ranks])
        self.capacity = self.rb().rpc_sync().capacity()
        
    def rb(self):
        idx = random.randint(0, len(self.replay_rrefs)-1)
        return self.replay_rrefs[idx]

  
    def test_clear(self):
        self.rb().rpc_sync().put(*list(range(self.capacity)))
        self.rb().rpc_sync().clear()
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), 0, 'clear'))

    def test_sample(self):
        num_data = self.capacity // 2
        self.rb().rpc_sync().put(*list(range(num_data)))

        # sample
        ans = self.rb().rpc_sync().sample(num_data)
        self.cases.append(('assertEqual', len(ans), num_data, 'sample'))

        # sample overflow
        ans = self.rb().rpc_sync().sample(self.capacity*2)
        self.cases.append(('assertEqual', len(ans), self.capacity*2, 'sample overflow'))

        # sample zero
        ans = self.rb().rpc_sync().sample(0)
        self.cases.append(('assertEqual', len(ans), 0, 'sample zero'))

        # sample empty
        self.rb().rpc_sync().clear()
        ans = self.rb().rpc_sync().sample(10)
        self.cases.append(('assertEqual', ans, None, 'sample empty'))
        
    def test_popn(self):
        self.rb().rpc_sync().clear()
        num_data, pop_size = self.capacity, self.capacity // 3
        self.rb().rpc_sync().put(*list(range(num_data)))
        self.rb().rpc_sync().popn(pop_size)
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data - pop_size, 'pop'))
        ans = self.rb().rpc_sync().sample((num_data - pop_size) * 2)
        ex_ans = [x for x in ans if x >= num_data - pop_size]
        self.cases.append(('assertEqual', len(ex_ans), 0, f'pop with error {ex_ans}'))

    def test_popnleft(self):
        self.rb().rpc_sync().clear()
        self.rb().rpc_sync().clear()
        num_data, pop_size = self.capacity, self.capacity // 3
        self.rb().rpc_sync().put(*list(range(num_data)))
        self.rb().rpc_sync().popnleft(pop_size)
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data - pop_size, 'pop'))
        ans = self.rb().rpc_sync().sample((num_data - pop_size) * 2)
        ex_ans = [x for x in ans if x < pop_size]
        self.cases.append(('assertEqual', len(ex_ans), 0, f'pop left with error {ex_ans}'))

   

class TestListReplayBuffer(unittest.TestCase):
    
    def setUp(self):
        file = tempfile.NamedTemporaryFile()
        args = [{'target': TestTarget}] + [{'target': ReplayBuffer}] * random.randint(1, 10)
        self.lanucher = mp.lanuch_async(
            manager=Manager,
            args = args,
            rpc = {
                'init_method': f'file://{file.name}',
                'world_size': len(args)
            },
            capacity=100
        )

    def tearDown(self) -> None:
        self.lanucher.join()


    def test_rb_list_append(self, test=False, tself=None):
        if not test: return

        # append a few
        num_data = random.randint(tself.capacity // 3, tself.capacity // 2)
        futures.wait_all([self.rb().rpc_async().append(i) for i in range(num_data)])
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data, 'put'))
        self.assertEqual()

        # put overflow
        self.rb().rpc_sync().put(*list(range(self.capacity)))
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), self.capacity, 'put overflow'))



if __name__ == '__main__':
    unittest.main()