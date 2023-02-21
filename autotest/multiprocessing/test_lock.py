import unittest
import time
from collections import defaultdict
import torzilla.multiprocessing as mp

class Manager(mp.Manager):
    def _start(self):
        kwargs = mp.current_target().kwargs()
        if kwargs['lock_type'] == 'manager':
            self.mutex = self.RWLock(kwargs['w_first'])
        else:
            self.mutex = mp.RWLock(kwargs['w_first'])
        self.value = self.Value('i', 0)
        self.barrier = self.Barrier(parties=mp.current_target().num_process())
        self.event = self.list()

    def _exit(self):
        self.results = {
            'value': self.value.value,
            'event': list(self.event)
        }

class Target(mp.Target):
    def _run(self):
        num_step = self.kwargs().get('num_step', 1)
        step_sleep = self.kwargs().get('step_sleep', 0)
        lock_sleep = self.kwargs().get('lock_sleep', 0)
        pre_sleep = self.kwargs().get('pre_sleep', 0)

        lock = self.manager().mutex.rlock() if self.who() == 'r' else self.manager().mutex.wlock()
        self.manager().barrier.wait()

        time.sleep(pre_sleep)
        for i in range(num_step):
            st = time.time()
            with lock:
                # print(f'{self.who()}-{self.index} enter lock {i}')
                self._step()
                time.sleep(lock_sleep)

            # print(f'{self.who()}-{self.index} exit lock {i}')
            self.event('step_time', time.time() - st)
            time.sleep(step_sleep)

    def who(self):
        return 'r' if isinstance(self, Reader) else 'w'
    
    def _step(self): pass

    def event(self, *args):
        self.manager().event.append((self.who(),) + args)


class Reader(Target):
    def _step(self):
        self.event('value', self.manager().value.value)


class Writer(Target):
    def _step(self):
        self.manager().value.value += 1
            

class TestRWLock(unittest.TestCase):
    def test_mp_rwlock_w_first(self):

        pre_reader = [
            {'target': Reader, 'pre_sleep': 0, 'lock_sleep': 0.1},
        ] * 3
        writer = [
            {'target': Writer, 'pre_sleep': 0.05, 'lock_sleep': 0.2},
            {'target': Writer, 'pre_sleep': 0.2, 'lock_sleep': 0.2},
        ]
        post_reader = [
            {'target': Reader, 'pre_sleep': 0.1, 'lock_sleep': 0.1},
        ] * 5
        subargs = pre_reader + writer + post_reader
        
        def check(results):
            gt_read_values = {
                0: len(pre_reader), 
                len(writer): len(post_reader)
            }

            # collect read values
            reads = defaultdict(int)
            for (who, ek, ev) in results['event']:
                if who != 'r' or ek != 'value': continue
                reads[ev] += 1
            
            for k, v in reads.items():
                self.assertTrue(k in gt_read_values, f'value {k} is error')
                self.assertEqual(v, gt_read_values[k], f'value count should be {gt_read_values[k]}')

        for lock_type in ['manager', 'mp']:
            results = mp.launch(
                args=subargs, manager=Manager, 
                w_first = True, lock_type = lock_type
            ).manager().results
            check(results)

    def test_mp_rwlock_r_first(self):
        pre_reader = [
            {'target': Reader, 'pre_sleep': 0, 'lock_sleep': 0.1},
        ] * 3
        writer = [
            {'target': Writer, 'pre_sleep': 0.05, 'lock_sleep': 0.2},
            {'target': Writer, 'pre_sleep': 0.2, 'lock_sleep': 0.2},
        ]
        post_reader = [
            {'target': Reader, 'pre_sleep':0.08, 'lock_sleep': 0.2},
            {'target': Reader, 'pre_sleep':0.2, 'lock_sleep': 0.2},
        ] * 5
        subargs = pre_reader + writer + post_reader

        def check(results):
            # collect read values
            for (who, ek, ev) in results['event']:
                if who != 'r' or ek != 'value': continue
                self.assertEqual(ev, 0, 'value count should be 0')
                

        for lock_type in ['manager', 'mp']:
            results = mp.launch(
                args=subargs, manager=Manager, 
                w_first = False, lock_type = lock_type
            ).manager().results
            check(results)


if __name__ == '__main__':
    unittest.main()