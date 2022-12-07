import unittest
import torzilla.multiprocessing as mp

class MainTarget(mp.Target):
    def _start(self):
        self.result = []

    def _exit(self):
        for _ in range(self.num_process()):
            self.result.append(self.manager().Q.get())
        

class Manager(mp.Manager):
    def _start(self):
        self.Q = self.Queue()


class SubTargetA(mp.Target):
    def _run(self):
        _target_A(self)


class SubTargetB(mp.Target):
    def _run(self):
        _target_B(self)


def _target_A(target):
    number = target.kwargs().get('number', target.num_process())
    target.manager().Q.put(('a', target.index(), int(number)))


def _target_B(target):
    number = target.kwargs().get('number', target.num_process())
    target.manager().Q.put(('b', target.index(), str(number)))

def _target_empty(target):
    pass


class TestTarget(unittest.TestCase):
    def test_mp_simple(self):
        mp.lanuch(
            num_process = 1,
            target = mp.Target,
        )
        mp.lanuch(
            num_process = 1,
            target = _target_empty,
        )

    def test_mp_target_class(self):
        ans = mp.lanuch(
            num_process=3,
            main = MainTarget,
            target = SubTargetA,
            manager=Manager,
        )
        self._common_check(ans.result)

    def test_mp_target_function(self):
        number = 100
        ans = mp.lanuch(
            num_process=5,
            main = MainTarget,
            manager = Manager,
            target = _target_A,
            number = number
        )
        self._common_check(ans.result, number)

    def test_mp_mix_target(self):
        ans = mp.lanuch(
            main = MainTarget,
            manager = Manager,
            args = [
                {'target': SubTargetA, 'number': 1},
                {'target': SubTargetB, 'number': 2},
                {'target': _target_A},
                {'target': _target_B},
            ]
        )
        n = ans.num_process()
        numbers = [1, '2', n, str(n)]
        for (who, idx, num) in ans.result:
            gt = numbers[idx]
            self.assertEqual(num, gt)

    def _common_check(self, result, number=None):
        number = number or len(result)
        for (who, idx, num) in result:
            if who == 'a':
                self.assertEqual(num, number)
            else:
                self.assertEqual(num, str(number))


if __name__ == '__main__':
    unittest.main()