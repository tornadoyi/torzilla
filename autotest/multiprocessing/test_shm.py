import unittest
import time
import torch
import torzilla.multiprocessing as mp

class MainProcess(mp.MainProcess):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._queue = mp.Queue()
        self._res_queue = mp.Queue()
        self._result = None
        mp.Manager.register('get_queue', callable=lambda: self._queue)
        mp.Manager.register('get_result_queue', callable=lambda: self._res_queue)

    @property
    def result(self): return self._result

    def _on_exit(self):
        self._result = self._res_queue.get()


class Subprocess(mp.Subprocess):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._queue = self.manager.get_queue()

    def _on_start(self):
        self._queue = self.manager.get_queue()
        if self.index == 0:
            self.consume()
        else:
            self.produce()

    def produce(self):
        SHAPE = (1024, 1024, 100)
        num_test_data = self.kwargs.get('num_test_data')
        for i in range(num_test_data):
            t = torch.rand(*SHAPE)
            t.share_memory_()
            data = {'data': t, 'time': time.time()}
            self._queue.put(data)

        for i in range(num_test_data):
            t = torch.rand(*SHAPE)
            data = {'data': t, 'time': time.time()}
            self._queue.put(data)

    def consume(self):
        num_test_data = self.kwargs.get('num_test_data')
        shared_cost = 0
        for i in range(num_test_data):
            d = self._queue.get()
            shared_cost += time.time() - d['time']
        non_shared_cost = 0
        for i in range(num_test_data):
            d = self._queue.get()
            non_shared_cost += time.time() - d['time']

        if non_shared_cost < shared_cost * 2:
            result = (False, f'shared time cost {shared_cost} is not much less than none shared time cost {non_shared_cost}')
        else:
            result = (True, None)
        self.manager.get_result_queue().put(result)

class TestSHM(unittest.TestCase):
    
    def test_mp_shm(self):
        proc = mp.lanuch(
            num_process=2,
            mainproc=MainProcess,
            subproc=Subprocess,
            num_test_data=10,
        )
        ok, err = proc.result
        if not ok: self.fail(err)

if __name__ == '__main__':
    unittest.main()