import unittest
import random
import torzilla.multiprocessing as mp


class TestCycledList(unittest.TestCase):

    def setUp(self):
        self.manager = mp.Manager()
        self.manager.start()
        self.clist = self.manager.clist(capacity=50)
        self.list = []

    def tearDown(self):
        self.manager.exit()
        
    def test_mp_clist_append(self):
        for i in range(10):
            self.clist.append(i)
            self.list.append(i)
        self.compare('append')

    def test_mp_clist_extend(self):
        datas = list(range(self.clist.capacity() // 3))
        self.clist.extend(datas)
        self.list.extend(datas)
        self.clist.extend(datas)
        self.list.extend(datas)
        self.compare('extend')

    def test_mp_clist_clear(self):
        datas = list(range(10))
        self.clist.extend(datas)
        self.list.extend(datas)
        self.clist.clear()
        self.list.clear()
        self.compare('clear')

    def test_mp_clist_pop(self):
        datas = list(range(10))
        n = random.randint(1, 5)
        self.clist.extend(datas)
        self.list.extend(datas)
        self.clist.pop(size=n)
        for _ in range(n):
            self.list.pop()
        self.compare('pop')

    def test_mp_clist_popleft(self):
        datas = list(range(10))
        n = random.randint(1, 5)
        self.clist.extend(datas)
        self.list.extend(datas)
        self.clist.popleft(size=n)
        self.list = self.list[n:]
        self.compare('popleft')

    def test_mp_clist_capacity(self):
        count, capacity = 0, self.clist.capacity()
        while count < capacity * 3:
            cnt = random.randint(capacity // 3, capacity // 2)
            count += cnt
            datas = [random.randint(1, 100) for _ in range(cnt)]
            self.clist.extend(datas)
            self.list.extend(datas)
            self.list = self.list[-capacity:]
            self.compare('capacity')
        
    def test_mp_clist_iterator(self):
        datas = list(range(10))
        self.clist.extend(datas)
        self.list.extend(datas)
        clist = list(self.clist)
        self.compare('iteator', clist=clist)

    def compare(self, msg, clist=None, list=None):
        clist, list = clist or self.clist, list or self.list
        self.assertEqual(len(clist), len(list), f'{msg}: length mismatch')
        for i in range(len(clist)):
            a, b = clist[i], list[i]
            self.assertEqual(a, b, f'{msg}: value mismatch')
    

if __name__ == '__main__':
    unittest.main()