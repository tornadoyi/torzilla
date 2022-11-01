import unittest
import time
import torzilla.multiprocessing as mp
from torzilla.rl.replay_buffer import ListBufferStore, ListReplayBuffer


class Manager(mp.Manager):
    def __init__(self, *args, capacity=100, send_qsize=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capacity = capacity
        self._send_qsize = send_qsize

    def _on_start(self, *args, **kwargs):
        self._store = ListBufferStore(capacity=self._capacity, send_qsize=self._send_qsize)
        self._store.start()

    def _on_exit(self, *args, **kwargs):
        self._store.exit()

    @property
    def store(self): return self._store

class MainProcss(mp.MainProcess):
    def _on_start(self):
        with self._create_manager():
            processes = self._spawn(join=False)
            self._run()
            for p in processes:
                p.join()

    def _run(self):
        pass


class ReplayBufferProcess(mp.Subprocess):
    def _on_start(self):
        store = self.manager.store
        self._buffer = ListReplayBuffer(store, self.kwargs['enable_flush'], self.kwargs['flush_timeout'])
        self._buffer.start()

    def _on_exit(self, *args, **kwargs):
        self._buffer.exit()
    
class TestListReplayBuffer(unittest.TestCase):
    
    def create_process(self, num_process, write_index, write_first):
        Q = mp.Queue()
        subproc_args = []
        for idx in range(num_process):
            if idx == write_index:
                args = {'Q': Q, 'subproc': Writer, 'pre_sleep': 0.1, 'post_sleep': 1.0, 'write_first': write_first}
            else:
                if idx < write_index:
                    args = {'Q': Q, 'subproc': Reader, 'pre_sleep': 0, 'post_sleep': 1.0, 'write_first': write_first}
                else:
                    args = {'Q': Q, 'subproc': Reader, 'pre_sleep': 0.3, 'post_sleep': 1.0, 'write_first': write_first}
                
            subproc_args.append(args)

        mp.lanuch(
            num_process=num_process, 
            manager=mp.SharedManager,
            subproc_args=subproc_args
        )
        return Q


if __name__ == '__main__':
    unittest.main()