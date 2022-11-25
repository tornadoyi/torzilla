import unittest
import random
import numpy as np
import torzilla.multiprocessing as mp

class MainProcess(mp.MainProcess):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.result = None

    def _on_start(self, *args, **kwargs):
        cases = self.manager.list()
        na, nb = self.kwargs['num_proc_a'], self.kwargs['num_proc_b']
        gear_a, gear_b = self.manager.gear_a, self.manager.gear_b

        # # test apply
        for i in range(10):
            a1, a2 = random.randint(0, 100), random.randint(0, 100)
            b1, b2 = random.randint(0, 100), random.randint(0, 100)
            gt_a, gt_b = a1 + a2, b1 + b2
            ra = gear_a.apply('add', a1, a2)
            rb = gear_b.apply('add', b1, b2)
            cases.append(('assertEqual', len(ra), na, 'count of result a'))
            cases.append(('assertEqual', len(rb), nb, 'count of result b'))
            for r in ra:
                cases.append(('assertEqual', r, gt_a, 'add result'))
            for r in rb:
                cases.append(('assertEqual', r, gt_b, 'add result'))

        # test apply async
        ars = []
        for i in range(10):
            a1, a2 = random.randint(0, 100), random.randint(0, 100)
            b1, b2 = random.randint(0, 100), random.randint(0, 100)
            gt_a, gt_b = a1 + a2, b1 + b2
            ra = gear_a.apply_async('add', a1, a2)
            rb = gear_b.apply_async('add', b1, b2)
            ars.append((ra, gt_a))
            ars.append((rb, gt_b))

        for (ar, gt) in ars:
            for r in ar.get():
                cases.append(('assertEqual', r, gt, 'add aync'))

        # test error
        ex = False
        try:
            gear_a.apply('add', 1, 1, 1)
        except:
            ex = True
            
        cases.append(('assertEqual', ex, True, 'add aync'))
        self.result = list(cases)

    def _on_exit(self):
        self.manager.close_gears()
        

class Manager(mp.Manager):
    def _on_start(self, *args, **kwargs):
        kwds = mp.Process.current().kwargs
        self.gear_a = self.Gear(kwds['num_proc_a'])
        self.gear_b = self.Gear(kwds['num_proc_b'])

    def close_gears(self, *args, **kwargs):
        self.gear_a.close()
        self.gear_b.close()


class Subprocess(mp.Subprocess):
    def _on_start(self):
        def dispatch(method, *args, **kwargs):
            return getattr(self, method)(*args, **kwargs)
        gear = self.get_gear()
        conn = gear.connect(dispatch)
        conn.join()

    def add(self, a, b): return a + b

class SubprocessA(Subprocess):
    def get_gear(self): 
        return self.manager.gear_a

class SubprocessB(Subprocess):
    def get_gear(self): 
        return self.manager.gear_b     
    
class TestGear(unittest.TestCase):
    
    def test_mp_gear(self):
        subproc_args = []
        na, nb = random.randint(3, 10), random.randint(3, 10)
        for _ in range(na):
            subproc_args.append(dict(subproc=SubprocessA))
        for _ in range(nb):
            subproc_args.append(dict(subproc=SubprocessB))
        proc = mp.lanuch(
            na + nb,
            mainproc=MainProcess,
            manager=Manager,
            subproc_args = subproc_args,
            num_proc_a = na,
            num_proc_b = nb,
        )

        for it in proc.result:
            getattr(self, it[0])(*it[1:])
    

if __name__ == '__main__':
    unittest.main()