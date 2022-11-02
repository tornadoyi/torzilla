import unittest
import time
import tempfile
import random
import torzilla.multiprocessing as mp
from torzilla.rl.replay_buffer import ListBufferStore, ListReplayBuffer
from torzilla import rpc

class Manager(mp.Manager):
    def __init__(self, *args, capacity=10, max_cache_size=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capacity = capacity
        self._max_cache_size = max_cache_size
        self.shared = None
        self.result = None

    @property
    def store(self): return self._store

    def _on_start(self, *args, **kwargs):
        self.shared = self.dict()
        self._store = ListBufferStore(capacity=self._capacity, max_cache_size=self._max_cache_size)
        self._store.start()

    def _on_exit(self, *args, **kwargs):
        self._store.exit()
        self.result = dict(self.shared)


class BaseTestProcess(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        shared = self.manager.shared

        # get all replays
        rank = rpc.get_worker_info().id
        replay_ranks = [info.id for info in rpc.get_worker_infos() if info.id != rank]
        futures = [rpc.rpc_async(str(r), mp.Process.current_rref) for r in replay_ranks]
        replay_rrefs = [fut.wait() for fut in futures]

        def rb():
            idx = random.randint(0, len(replay_rrefs)-1)
            return replay_rrefs[idx]

        def _wait(size, timeout):
            end_time = time.time() + timeout
            while time.time() < end_time:
                if rb().rpc_sync().size() == size: break
                time.sleep(timeout / 10)

        # send half
        capacity = rb().rpc_sync().capacity()
        for i in range(capacity // 2):
            rb().rpc_sync().put(i)
        _wait(capacity // 2, 3)
        shared['send half'] = (rb().rpc_sync().size(), capacity // 2)

        # send all
        for i in range(capacity):
            rb().rpc_sync().put(i)
        _wait(capacity, 3)
        shared['send all'] = (rb().rpc_sync().size(), capacity)

        # sample half
        datas = rb().rpc_sync().sample(capacity // 2)
        shared['sample half'] = (len(datas), capacity // 2)

        # sample all
        datas = rb().rpc_sync().sample(capacity)
        shared['sample all'] = (len(datas), capacity)

        # sample overflow
        datas = rb().rpc_sync().sample(capacity * 2)
        shared['sample all'] = (datas, None)

        # close
        self.manager.store.close()



class ReplayBufferProcess(mp.Subprocess):
    def _on_start(self):
        store = self.manager.store
        self._buffer = ListReplayBuffer(store, self.kwargs['enable_flush'])
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
        except:
            return None


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
        for k, (v, gt) in proc.manager.result.items():
            self.assertEqual(v, gt, f'{k} check fail')



if __name__ == '__main__':
    unittest.main()