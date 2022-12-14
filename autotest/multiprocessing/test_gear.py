import unittest
import random
import numpy as np
import torzilla.multiprocessing as mp


class Manager(mp.Manager):
    def _start(self):
        kwds = mp.current_target().kwargs()
        self.gear_a = self.Gear(kwds['num_a'])
        self.gear_b = self.Gear(kwds['num_b'])

    def close_gear(self):
        self.gear_a.close()
        self.gear_b.close()


class Target(mp.Target):
    def _run(self):
        def dispatch(method, *args, **kwargs):
            return getattr(self, method)(*args, **kwargs)
        gear = self.get_gear()
        gear.connect(dispatch, slot=self.kwargs()['slot']).join()

    def add(self, a, b): return a + b

class TargetA(Target):
    def get_gear(self): 
        return self.manager().gear_a

class TargetB(Target):
    def get_gear(self): 
        return self.manager().gear_b     
    
class TestGear(unittest.TestCase):

    def setUp(self) -> None:
        args = []
        na, nb = random.randint(3, 10), random.randint(3, 10)
        for i in range(na):
            args.append(dict(target=TargetA, slot=i))
        for i in range(nb):
            args.append(dict(target=TargetB, slot=i))
        self.result = mp.lanuch_async(
            manager=Manager,
            args = args,
            num_a = na,
            num_b = nb,
        )

    def tearDown(self) -> None:
        mp.current_target().manager().close_gear()
        self.result.get()

    def test_mp_gear_apply(self):
        manager = mp.current_target().manager()
        gear_a, gear_b = manager.gear_a, manager.gear_b
        for i in range(10):
            a1, a2 = random.randint(0, 100), random.randint(0, 100)
            b1, b2 = random.randint(0, 100), random.randint(0, 100)
            gt_a, gt_b = a1 + a2, b1 + b2
            ra = gear_a.apply('add', args=(a1, a2))
            rb = gear_b.apply('add', args=(b1, b2))

            self.assertEqual(len(ra), gear_a.connections())
            self.assertEqual(len(rb), gear_b.connections())

            for r in ra:
                self.assertEqual(r, gt_a)
            for r in rb:
                self.assertEqual(r, gt_b)

    def test_mp_gear_apply_async(self):
        manager = mp.current_target().manager()
        gear_a, gear_b = manager.gear_a, manager.gear_b
        for i in range(10):
            a1, a2 = random.randint(0, 100), random.randint(0, 100)
            b1, b2 = random.randint(0, 100), random.randint(0, 100)
            gt_a, gt_b = a1 + a2, b1 + b2
            ra = gear_a.apply_async('add', args=(a1, a2))
            rb = gear_b.apply_async('add', args=(b1, b2))

            ra, rb = ra.get(), rb.get()

            self.assertEqual(len(ra), gear_a.connections())
            self.assertEqual(len(rb), gear_b.connections())

            for r in ra:
                self.assertEqual(r, gt_a)
            for r in rb:
                self.assertEqual(r, gt_b)

    def test_mp_gear_apply_to(self):
        manager = mp.current_target().manager()
        gear_a, gear_b = manager.gear_a, manager.gear_b

        rs = gear_a.apply('add', args=(1, 1), to=(0, 1))
        for r in rs:
            self.assertEqual(r, 2)


if __name__ == '__main__':
    unittest.main()