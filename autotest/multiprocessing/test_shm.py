import unittest
import time
import torch
import torzilla.multiprocessing as mp

class Manager(mp.Manager):

    def _start(self):
        self.Q = self.Queue()
        self.R = self.Queue()
        self.result = None

    def _exit(self):
        self.result = self.R.get()


class Producer(mp.Target):
    def _run(self):
        Q = self.manager().Q
        SHAPE = (1024, 1024, 10)
        num_test_data = self.kwargs().get('num_test_data')
        for i in range(num_test_data):
            t = torch.rand(*SHAPE)
            t.share_memory_()
            data = {'data': t, 'time': time.time()}
            Q.put(data)

        for i in range(num_test_data):
            t = torch.rand(*SHAPE)
            data = {'data': t, 'time': time.time()}
            Q.put(data)


class Consumer(mp.Target):
    def _run(self):
        Q = self.manager().Q
        num_test_data = self.kwargs().get('num_test_data')
        shared_cost = 0
        for i in range(num_test_data):
            d = Q.get()
            shared_cost += time.time() - d['time']
        non_shared_cost = 0
        for i in range(num_test_data):
            d = Q.get()
            non_shared_cost += time.time() - d['time']

        if non_shared_cost < shared_cost * 2:
            result = (False, f'shared time cost {shared_cost} is not much less than none shared time cost {non_shared_cost}')
        else:
            result = (True, None)
        self.manager().R.put(result)


class TestSHM(unittest.TestCase):
    
    def test_mp_shm(self):
        proc = mp.lanuch(
            manager = Manager,
            args = [
                {'target': Producer},
                {'target': Consumer},
            ],
            num_test_data = 5,
        )
        ok, err = proc.manager().result
        if not ok: self.fail(err)

if __name__ == '__main__':
    unittest.main()