import unittest
import random
import torzilla.multiprocessing as mp

class MainProcess(mp.MainProcess):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result = []

    @property
    def result(self): return self._result

    def _on_exit(self):
        queue = self.manager.get_queue()
        for _ in range(self.num_process):
            self._result.append(queue.get())
        

class Manager(mp.Manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = mp.Queue()
    
    def get_queue(self): return self._queue


class SubprocessA(mp.Subprocess):
    def _on_start(self):
        number = self.kwargs['number']
        queue = self.manager.get_queue()
        queue.put((self.index, number))

class SubprocessB(mp.Subprocess):
    def _on_start(self):
        number = self.kwargs['number']
        queue = self.manager.get_queue()
        queue.put((self.index, str(number)))

    
class TestProcess(unittest.TestCase):
    
    def test_common(self):
        mp.lanuch(num_process=5)

    def test_shared_process(self):
        gt = 100
        proc = mp.lanuch(
            num_process=5,
            mainproc=MainProcess,
            subproc=SubprocessA,
            manager=Manager,
            number=gt
        )
        for (idx, num) in proc.result:
            self.assertEqual(num, gt)

    def test_independ_process(self):
        subproc_args = []
        for _ in range(10):
            number = random.randint(0, 100)
            if number < 50:
                args = {'subproc': SubprocessA, 'number': number}
            else:
                args = {'subproc': SubprocessB, 'number': number}
            subproc_args.append(args)
        
        proc = mp.lanuch(
            num_process=len(subproc_args),
            mainproc=MainProcess,
            manager=Manager,
            subproc_args=subproc_args,
        )
        for (idx, num) in proc.result:
            args = subproc_args[idx]
            gt = args['number']
            if args['subproc'] is SubprocessB: gt = str(gt)
            self.assertEqual(num, gt)

if __name__ == '__main__':
    unittest.main()