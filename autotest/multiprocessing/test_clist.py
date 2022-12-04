import unittest
import random
import math
import numpy as np
import torzilla.multiprocessing as mp
from torzilla import NotExist


def _catch(func):
        try:
            return func()
        except Exception as e:
            # print(e)
            return e

class TestCycledList(unittest.TestCase):
    NUM_TEST = 10

    def setUp(self):
        self.manager = mp.Manager()
        self.manager.start()
        self.reset_list()

    def tearDown(self):
        self.manager.exit()
    
    def test_mp_clist___add__(self, n=NUM_TEST):
        for _ in range(n):
            other = self.randlist()
            self.compare(self.clist + other, self.truncate(self.list + other))

    def test_mp_clist___contains__(self, n=NUM_TEST):
        values = self.randlist(size=n)
        self.compare(
            [v in self.clist for v in values],
            [v in self.list for v in values],
        )
            
    def test_mp_clist___copy__(self, n=NUM_TEST):
        for _ in range(n):
            self.compare(self.clist.copy(), self.list.copy())

    def test_mp_clist___getitem__(self, n=NUM_TEST):
        while len(self.clist) < 100:
            self.test_mp_clist_extend()
        cap = self.clist.capacity()
        # index
        for idx in self.randlist(high=self.clist.capacity() * 2, size=n):
            self.compare(
                _catch(lambda: self.clist[idx]), 
                _catch(lambda: self.list[idx]),
            )
        # list
        lows = self.randlist(low=-3, high=math.ceil(cap*1.1), size=n)
        for low in lows:
            size = random.randint(100, 200)
            high = random.randint(low+1, math.ceil(cap*1.1))
            idxes = self.randlist(low=low, high=high, size=size)
            
            self.compare(
                _catch(lambda: self.clist[idxes]), 
                _catch(lambda: [self.list[idx] for idx in idxes]),
            )
        # slice
        starts = self.randlist(low=math.floor(-cap*1.1), high=math.ceil(cap*1.1), size=n)
        stops = self.randlist(low=math.floor(-cap*1.1), high=math.ceil(cap*1.1), size=n)
        steps = self.randlist(low=-3, high=3, size=n, reject=[0])
        for i in range(len(starts)):
            s = slice(starts[i], stops[i], steps[i])
            self.compare(self.clist[s], self.list[s])

    def test_mp_clist___iadd__(self, n=NUM_TEST):
        for _ in range(n):
            values = self.randlist()
            self.clist += values
            self.list += values
            self.list = self.truncate(self.list)
            self.compare()

    def test_mp_clist___imul__(self, n=NUM_TEST):
        for _ in range(n):
            size = random.randint(0, 10)
            self.clist *= size
            self.list *= size
            self.list = self.truncate(self.list)
            self.compare()

    def test_mp_clist___len__(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare()
        
    def test_mp_clist___mul__(self, n=NUM_TEST):
        for _ in range(n):
            size = random.randint(0, 10)
            self.compare(self.clist * size, self.truncate(self.list * size))

    def test_mp_clist___repr__(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(str(self.clist), str(self.list))

    def test_mp_clist___reversed__(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(list(reversed(self.clist)), list(reversed(self.list)))

    def test_mp_clist___rmul__(self, n=NUM_TEST):
        for _ in range(n):
            size = random.randint(0, 10)
            self.compare(size * self.clist, self.truncate(size * self.list))

    def test_mp_clist___setitem__(self, n=NUM_TEST):
        while len(self.clist) < 100:
            self.test_mp_clist_extend()
        cap = self.clist.capacity()

        def _set(list, idx, data):
            list[idx] = data
        def _sets(list, idxes, datas):
            for i, idx in enumerate(idxes):
                list[idx] = datas[i]

        # index
        for idx in self.randlist(high=self.clist.capacity() * 2, size=n):
            data = random.randint(0, 100)
            self.compare(
                _catch(lambda: _set(self.clist, idx, data)), 
                _catch(lambda: _set(self.list, idx, data)),
            )
        # list
        lows = self.randlist(low=-3, high=math.ceil(cap*1.1), size=n)
        for low in lows:
            size = random.randint(100, 200)
            high = random.randint(low+1, math.ceil(cap*1.1))
            idxes = self.randlist(low=low, high=high, size=size)
            datas = self.randlist(low=0, high=100, size=size)
            self.compare(
                _catch(lambda: _set(self.clist, idxes, datas)), 
                _catch(lambda: _sets(self.list, idxes, datas)),
            )
        # slice
        lsize = len(self.clist)
        size = random.randint(100, 200)
        starts = self.randlist(low=-lsize, high=lsize-1, size=n)
        stops = self.randlist(low=-lsize, high=lsize-1, size=n)
        steps = self.randlist(low=-3, high=3, size=n, reject=[0])
        for i in range(len(starts)):
            s = slice(starts[i], stops[i], steps[i])
            v_idxes = list(range(lsize)[s])
            datas = self.randlist(0, 100, size=len(v_idxes))
            self.clist[s] = datas
            self.list[s] = datas
            self.compare()

    def test_mp_clist___iter__(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(
                [x for x in self.clist],
                [x for x in self.list],
            )

    def test_mp_clist_append(self, n=NUM_TEST):
        for i in range(n):
            self.clist.append(i)
            self.list.append(i)
        self.list = self.truncate(self.list)
        self.compare()

    def test_mp_clist_appendleft(self, n=NUM_TEST):
        for i in range(n):
            self.clist.appendleft(i)
            self.list = [i] + self.list
        self.list = self.truncateleft(self.list)
        self.compare()

    def test_mp_clist_clear(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.clist.clear()
            self.list.clear()
            self.compare()

    def test_mp_clist_copy(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(self.clist.copy(), self.list.copy())

    def test_mp_clist_extend(self, n=NUM_TEST):
        for _ in range(n):
            other = self.randlist()
            self.clist.extend(other)
            self.list.extend(other)
            self.list = self.truncate(self.list)
            self.compare()

    def test_mp_clist_extendleft(self, n=NUM_TEST):
        for _ in range(n):
            other = self.randlist()
            self.clist.extendleft(other)
            self.list = self.truncateleft(other[-1::-1] + self.list)
            self.compare()

    def test_mp_clist_count(self, n=NUM_TEST):
        for idx in self.randlist(0, len(self.clist), size=n):
            v = self.clist[idx]
            self.compare(self.clist.count(v), self.list.count(v))

    def test_mp_clist_index(self, n=NUM_TEST):
        size = len(self.clist)
        for idx in self.randlist(0, size, size=n):
            v = self.clist[idx]
            start = random.randint(-size, size)
            stop = random.randint(start, size)
            self.compare(
                _catch(lambda: self.clist.index(v, start, stop)), 
                _catch(lambda: self.list.index(v, start, stop)),
                f'v: {v} start: {start} stop: {stop}'
            )
    
    def test_mp_clist_pop(self, n=NUM_TEST):
        for _ in range(n):
            self.compare(
                _catch(lambda: self.clist.pop()),
                _catch(lambda: self.list.pop()),
            )

    def test_mp_clist_popleft(self, n=NUM_TEST):
        def _popleftlist():
            if len(self.list) == 0:
                raise ValueError('not enough')
            r = self.list[0]
            self.list = self.list[1:]
            return r

        for _ in range(n):
            self.compare(
                _catch(lambda: self.clist.popleft()),
                _catch(lambda: _popleftlist()),
            )

    def test_mp_clist_popn(self, n=NUM_TEST):
        def _popnlist(n):
            if n == 0: return []
            if len(self.list) < n:
                raise ValueError('not enough')
            r = self.list[-1:-n-1:-1]
            self.list = self.list[0:-n]
            return r

        for cnt in self.randlist(0, len(self.clist), size=n):
            self.reset_list()
            self.compare(
                _catch(lambda: self.clist.popn(cnt)),
                _catch(lambda: _popnlist(cnt)),
                f'n: {cnt}'
            )

    def test_mp_clist_popnleft(self, n=NUM_TEST):
        def _popnleftlist(n):
            if n == 0: return []
            if len(self.list) < n:
                raise ValueError('not enough')
            r = self.list[0:n]
            self.list = self.list[n:]
            return r

        for cnt in self.randlist(0, len(self.clist), size=n):
            self.reset_list()
            self.compare(
                _catch(lambda: self.clist.popnleft(cnt)),
                _catch(lambda: _popnleftlist(cnt)),
                f'n: {cnt}'
            )

    def test_mp_clist_reverse(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(
                self.clist.reverse(),
                self.list.reverse(),
            )

    def test_mp_clist_sort(self, n=NUM_TEST):
        for _ in range(n):
            self.reset_list()
            self.compare(
                self.clist.sort(),
                self.list.sort(),
            )

    def compare(self, a=None, b=None, msg=''):
        def _format_info():
            last_state = self.last_clist._state()
            state = self.clist._state()
            return '\n' + '\n'.join([x for x in [
                f'[begin]',
                f'clist: {self.last_clist}',
                f'M    : {last_state["M"]}',
                f'head : {last_state["head"]}' ,
                f'size : {last_state["size"]}' ,
                f'list : {self.last_list}',
                f'',
                f'[end]',
                f'clist: {self.clist}',
                f'M    : {state["M"]}',
                f'head : {state["head"]}',
                f'size : {state["size"]}',
                f'list : {self.list}',
                f'',
                f'[return]',
                f'a    : {a}',
                f'b    : {b}',
                f'msg  : {msg}',
            ] if x is not None])

        # compare list
        self.assertEqual(len(self.clist), len(self.list), f'list length mismatch {_format_info()}')
        for i in range(len(self.clist)):
            self.assertEqual(self.clist[i], self.list[i], f'list value mismatch {_format_info()}')

        # compare input
        self.assertEqual(
            hasattr(a, '__iter__'), 
            hasattr(b, '__iter__'), 
            f'input iter mismatch {_format_info()})'
        )
        if not hasattr(a, '__iter__'):
            a, b = [a], [b]
        self.assertEqual(len(a), len(b), f'input length mismatch')
        for i in range(len(a)):
            x, y = a[i], b[i]
            # exception
            x_ex, y_ex = isinstance(x, Exception), isinstance(y, Exception)
            if x_ex + y_ex > 0:
                self.assertTrue(x_ex - y_ex == 0, f'value mismatch {_format_info()}')
            else:
                self.assertEqual(x, y, f'value mismatch {_format_info()}')

        self.screenshot()

    def randlist(self, low=0, high=100, size=None, reject=[]):
        size = random.randint(0, self.clist.capacity()) if size is None else size
        ans = []
        while len(ans) != size:
            need = size - len(ans)
            ans += [int(x) for x in np.random.randint(low, high, need) if x not in reject]
        return ans

    def truncate(self, list):
        cap = self.clist.capacity()
        return list[-cap:]

    def truncateleft(self, list):
        cap = self.clist.capacity()
        return list[:cap]

    def reset_list(self):
        self.clist = self.manager.clist(capacity=120)
        self.list = []

        # random init datas
        other = self.randlist(reject=list(range(self.clist.capacity() // 3)))
        # other = list(range(50))
        self.clist.extend(other)
        self.list.extend(other)
        self.screenshot()

    def screenshot(self):
        self.last_clist = self.clist.copy()
        self.last_list = self.list.copy()


if __name__ == '__main__':
    unittest.main()