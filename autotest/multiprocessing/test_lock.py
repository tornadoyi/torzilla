import unittest
import time
import torzilla.multiprocessing as mp

class Subprocess(mp.Subprocess):
    def _get_shared_data(self, write_first):
        with self.manager.shared_lock:
            data = self.manager.shared_data
            if len(data) == 0:
                data['lock'] = self.manager.RWLock(write_first=write_first)
                data['value'] = self.manager.Value('i', 0)
        return data['lock'], data['value']

class Reader(Subprocess):
    def _on_start(self):
        pre_sleep, post_sleep = self.kwargs['pre_sleep'], self.kwargs['post_sleep']
        Q = self.kwargs['Q']
        lock, value = self._get_shared_data(self.kwargs['write_first'])
        time.sleep(pre_sleep)
        with lock.reader_lock():
            Q.put((self.index, value.value))
            time.sleep(post_sleep)

class Writer(Subprocess):
    def _on_start(self):
        pre_sleep, post_sleep = self.kwargs['pre_sleep'], self.kwargs['post_sleep']
        time.sleep(pre_sleep)
        lock, value = self._get_shared_data(self.kwargs['write_first'])
        with lock.writer_lock():
            value.value = self.index
    
class TestProcess(unittest.TestCase):
    
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


    def test_mp_rwlock_write_first(self):
        num_process, write_index = 7, 3
        Q = self.create_process(num_process, write_index, write_first=True)
        for _ in range(Q.qsize()):
            idx, value = Q.get()
            if idx < write_index:
                self.assertEqual(value, 0)
            else:
                self.assertEqual(value, write_index)

    def test_mp_rwlock_read_first(self):
        num_process, write_index = 7, 3
        Q = self.create_process(num_process, write_index, write_first=False)
        for _ in range(Q.qsize()):
            idx, value = Q.get()
            self.assertEqual(value, 0)

if __name__ == '__main__':
    unittest.main()