import unittest
from torch import futures
import tempfile
import random
import torzilla.multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer
from torzilla import rpc

class Manager(mp.Manager):
    def __init__(self, *args, capacity=10, max_cache_size=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capacity = capacity
        self._max_cache_size = max_cache_size
        self.result = None

    def _on_start(self, *args, **kwargs):
        self._cases = self.list()
        self._buffer = ListReplayBuffer(capacity=self._capacity, max_cache_size=self._max_cache_size)
        self._buffer.start()

    def _on_exit(self, *args, **kwargs):
        self._buffer.exit()
        self.result = list(self._cases)


class BaseTestProcess(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        self.cases = self.manager._cases

        # get all replays
        rank = rpc.get_worker_info().id
        replay_ranks = [info.id for info in rpc.get_worker_infos() if info.id != rank]
        self.replay_rrefs = futures.wait_all([rpc.rpc_async(str(r), mp.Process.current_rref) for r in replay_ranks])
        self.capacity = self.rb().rpc_sync().capacity()

        self.test_put()
        self.test_clear()
        self.test_sample()
        self.test_pop()
        self.test_popleft()
        
    def rb(self):
        idx = random.randint(0, len(self.replay_rrefs)-1)
        return self.replay_rrefs[idx]

    def test_put(self):
        # put
        num_data = self.capacity // 2
        futures.wait_all([self.rb().rpc_async().put(i) for i in range(num_data)])
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data, 'put'))

        # put overflow
        self.rb().rpc_sync().put(*list(range(self.capacity)))
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), self.capacity, 'put overflow'))
    
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
        self.cases.append(('assertEqual', ans, None, 'sample zero'))

        # sample empty
        self.rb().rpc_sync().clear()
        ans = self.rb().rpc_sync().sample(10)
        self.cases.append(('assertEqual', ans, None, 'sample empty'))
        
    def test_pop(self):
        self.rb().rpc_sync().clear()
        num_data, pop_size = self.capacity, self.capacity // 3
        self.rb().rpc_sync().put(*list(range(num_data)))
        self.rb().rpc_sync().pop(pop_size)
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data - pop_size, 'pop'))
        ans = self.rb().rpc_sync().sample((num_data - pop_size) * 2)
        ex_ans = [x for x in ans if x >= num_data - pop_size]
        self.cases.append(('assertEqual', len(ex_ans), 0, f'pop with error {ex_ans}'))

    def test_popleft(self):
        self.rb().rpc_sync().clear()
        self.rb().rpc_sync().clear()
        num_data, pop_size = self.capacity, self.capacity // 3
        self.rb().rpc_sync().put(*list(range(num_data)))
        self.rb().rpc_sync().popleft(pop_size)
        self.cases.append(('assertEqual', self.rb().rpc_sync().size(), num_data - pop_size, 'pop'))
        ans = self.rb().rpc_sync().sample((num_data - pop_size) * 2)
        ex_ans = [x for x in ans if x < pop_size]
        self.cases.append(('assertEqual', len(ex_ans), 0, f'pop left with error {ex_ans}'))


class ReplayBufferProcess(mp.Subprocess):
    def _on_start(self):
        master = self.manager._buffer
        self._buffer = ListReplayBuffer(master=master)
        self._buffer.start()

    def _on_exit(self, *args, **kwargs):
        self._buffer.exit()

    def size(self):
        return len(self._buffer)

    def capacity(self):
        return self._buffer.capacity

    def put(self, *args, **kwargs):
        return self._buffer.put(*args, **kwargs)

    def sample(self, *args, **kwargs):
        try:
            return self._buffer.sample(*args, **kwargs)
        except Exception as e:
            return None

    def clear(self):
        self._buffer.clear()

    def pop(self, *args, **kwargs):
        self._buffer.pop(*args, **kwargs)

    def popleft(self, *args, **kwargs):
        self._buffer.popleft(*args, **kwargs)


class TestListReplayBuffer(unittest.TestCase):
    
    def test_rb_list(self):
        num_process = 5
        subproc_args = [dict(subproc=BaseTestProcess)]
        for _ in range(num_process-1):
            subproc_args.append(dict(
                subproc=ReplayBufferProcess,
                enable_flush=True,
            ))

        file = tempfile.NamedTemporaryFile()
        proc = mp.lanuch(
            manager=Manager,
            subproc_args=subproc_args,
            rpc = {
                'init_method': f'file://{file.name}'
            },
        )
        for it in proc.manager.result:
            getattr(self, it[0])(*it[1:])



if __name__ == '__main__':
    unittest.main()