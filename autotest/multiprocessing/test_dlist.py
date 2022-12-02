import unittest
import random
from collections import deque
import torzilla.multiprocessing as mp


class TestDictdList(unittest.TestCase):

    def setUp(self):
        self.manager = mp.Manager()
        self.manager.start()
        self.dlist = self.manager.dlist()
        self.deque = deque()

    def tearDown(self):
        self.manager.exit()
        
    def test_mp_dlist_append(self):
        steps = random.randint(5, 10)
        for i in range(steps):
            self.dlist.append(i)
            self.deque.append(i)
        self.compare('append')

    def test_mp_dlist_appendleft(self):
        steps = random.randint(5, 10)
        for i in range(steps):
            self.dlist.appendleft(i)
            self.deque.appendleft(i)
        self.compare('appendleft')

    def test_mp_dlist_extend(self):
        size = random.randint(10, 50)
        datas = [random.randint(1, 10) for _ in range(size)]
        self.dlist.extend(datas)
        self.deque.extend(datas)
        self.compare('extend')

    def test_mp_dlist_extendleft(self):
        size = random.randint(10, 50)
        datas = [random.randint(1, 10) for _ in range(size)]
        self.dlist.extendleft(datas)
        self.deque.extendleft(datas)
        self.compare('extendleft')

    def test_mp_dlist_clear(self):
        datas = list(range(10))
        self.dlist.extend(datas)
        self.deque.extend(datas)
        self.dlist.clear()
        self.deque.clear()
        self.compare('clear')

    def test_mp_dlist_pop(self):
        datas = list(range(10))
        n = random.randint(1, 5)
        self.dlist.extend(datas)
        self.deque.extend(datas)
        self.dlist.pop(size=n)
        for _ in range(n):
            self.deque.pop()
        self.compare('pop')

    def test_mp_dlist_popleft(self):
        datas = list(range(10))
        n = random.randint(1, 5)
        self.dlist.extend(datas)
        self.deque.extend(datas)
        self.dlist.popleft(size=n)
        for _ in range(n):
            self.deque.popleft()
        self.compare('popleft')
        
    def test_mp_dlist_iterator(self):
        datas = list(range(10))
        self.dlist.extend(datas)
        self.deque.extend(datas)
        dlist = [v for _, v in self.dlist.items()]
        self.compare('iteator', dlist=dlist)

    def compare(self, msg, dlist=None, deque=None):
        dlist, deque = dlist or self.dlist, deque or self.deque
        self.assertEqual(len(dlist), len(deque), f'{msg}: length mismatch')
        i = 0
        for key in dlist:
            a, b = dlist[key], deque[i]
            i += 1
            self.assertEqual(a, b, f'{msg}: value mismatch')
            
    

if __name__ == '__main__':
    unittest.main()