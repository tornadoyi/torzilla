import numbers
from operator import itemgetter


class CycledList(object):
    def __init__(self, capacity):
        self._M = [None] * capacity
        self._head = capacity
        self._size = 0

    @property
    def _tail(self):
        cap = self.capacity()
        if cap == 0:
            return 0
        return (self._head + self._size) % self.capacity()

    def __add__(self, other):
        o = self.copy()
        o.extend(other)
        return o

    def __contains__(self, value):
        for x in self:
            if x == value:
                return True
        return False

    def __copy__(self):
        o = CycledList(self.capacity())
        o._M = self._M.copy()
        o._head = self._head
        o._size = self._size
        return o

    def __getitem__(self, key):
        M, h, t = self._M, self._head, self._tail
        size, cap = len(self), len(M)

        def _norm_index(index):
            if index >= size or index < -size:
                raise IndexError(f'list index out of range, index: {index} size: {size}')
            return (t + index) % cap if index < 0 else (h + index) % cap

        def _index(index):
            return M[_norm_index(index)]

        def _list(indexes):
            idxes = [_norm_index(index) for index in indexes]
            if len(idxes) == 0: return []
            getter = itemgetter(*idxes)
            r = getter(M)
            return [r] if len(idxes) == 1 else list(r)

        def _slice(key):
            start, stop, step = key.start, key.stop, key.step
            r_indexes = range(len(self))[start:stop:step]
            idxes = [_norm_index(index) for index in r_indexes]
            if len(idxes) == 0: return []
            getter = itemgetter(*idxes)
            r = getter(M)
            return [r] if len(idxes) == 1 else list(r)
            
        # dispatch
        if isinstance(key, numbers.Number):
            return _index(key)
        elif hasattr(key, '__iter__'):
            return _list(key)
        elif isinstance(key, slice):
            return _slice(key)
        else:
            raise TypeError(f'invalid key type {type(key)}, expected: int, slice, list')

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __imul__(self, size):
        other = self[:] * size
        self.clear()
        self.extend(other)
        return self

    def __len__(self):
        return self._size

    def __mul__(self, size):
        o = CycledList(self.capacity())
        o.extend(self[:] * size)
        return o

    def __repr__(self):
        return str(self[:])

    def __reversed__(self):
        M, t, cap = self._M, self._tail, self.capacity()
        t_1 = t - 1
        for i in range(len(self)):
            yield M[(t_1 - i) % cap]

    def __rmul__(self, size):
        return self.__mul__(size)

    def __setitem__(self, key, value):
        M, h, t = self._M, self._head, self._tail
        size, cap = len(self), len(M)

        def _norm_index(index):
            if index >= size or index < -size:
                raise IndexError(f'list index out of range, index: {index} size: {size}')
            return (t + index) % cap if index < 0 else (h + index) % cap

        def _index(index, value):
            M[_norm_index(index)] = value

        def _list(indexes, value):
            for i, index in enumerate(indexes):
                M[_norm_index(index)] = value[i]

        def _slice(key, value):
            start, stop, step = key.start, key.stop, key.step
            r_indexes = range(len(self))[start:stop:step]
            for i, index in enumerate(r_indexes):
                M[_norm_index(index)] = value[i]
            
        # dispatch
        if isinstance(key, numbers.Number):
            return _index(key, value)
        elif hasattr(key, '__iter__'):
            return _list(key, value)
        elif isinstance(key, slice):
            return _slice(key, value)
        else:
            raise TypeError(f'invalid key type {type(key)}, expected: int, slice, list')

    def __iter__(self):
        M, h, cap = self._M, self._head, self.capacity()
        for i in range(len(self)):
            yield M[(h + i) % cap]

    def append(self, value):
        return self._extend([value], dir=1)

    def appendleft(self, value):
        return self._extend([value], dir=-1)
    
    def capacity(self):
        return len(self._M)

    def clear(self):
        self._M = [None] * self.capacity()
        self._head = self.capacity()
        self._size = 0

    def copy(self):
        return self.__copy__()

    def count(self, value):
        c = 0
        for v in self:
            if v != value:
                continue
            c += 1
        return c

    def extend(self, values):
        return self._extend(values, dir=1)

    def extendleft(self, values):
        return self._extend(values, dir=-1)

    def index(self, value, start=0, stop=None):
        M, h, cap = self._M, self._head, self.capacity()
        stop = len(self) if stop is None else stop
        r = range(len(self))[start:stop]
        for i in r:
            if M[(h + i) % cap] == value:
                return i
        raise ValueError(f'{value} is not in list')

    def pop(self):
        return self._popn(1, dir=1)[0]
            
    def popleft(self):
        return self._popn(1, dir=-1)[0]

    def popn(self, n):
        return self._popn(n, dir=1)
    
    def popnleft(self, n):
        return self._popn(n, dir=-1)

    def reverse(self):
        self[:] = list(reversed(self))[:]
        return None

    def sort(self, *, key=None, reverse=False):
        self[:] = sorted(self, key=key, reverse=reverse)[:]
        return None

    def _extend(self, values, dir):
        # check
        r_dsize = len(values)
        if r_dsize == 0: return

        #  truncate values to keep value size equals to capacity
        M, h, t, cap = self._M, self._head, self._tail, self.capacity()
        offset = 0
        if r_dsize > cap:
            values = values[-cap:]
            offset = r_dsize % cap
        dsize = len(values)

        # set final size
        self._size += r_dsize
        self._size = cap if self._size > cap else self._size
    
        # left
        if dir < 0:
            h = (h - 1 - offset) % cap
            rh = cap - h
            if h - dsize < -1:
                M[-rh:-cap-1:-1] = values[:-rh+cap+1]
                M[-1:-dsize-rh+cap:-1] = values[-rh+cap+1:]
            else:
                M[-rh:-rh-dsize:-1] = values[:]

            h = (h - dsize + 1) % cap
            self._head = h

        # right
        else:
            t = (t + offset) % cap
            if t + dsize > cap:
                M[t:cap] = values[:cap-t]
                M[0:dsize-cap+t] = values[cap-t:]
            else:
                M[t:t+dsize] = values[:]
            
            t = (t + dsize) % cap
            self._head = (t - self._size) % cap     # self._tail = t

    def _popn(self, n, dir):
        size = len(self)
        if n > size:
            raise ValueError(f'exceed current size {size}, n: {n}')
        if n == 0:
            return []

        # set final size
        M, h, t, cap = self._M, self._head, self._tail, self.capacity()

        if n == size:
            r = self[:] if dir < 0 else self[-1:-size-1:-1]
            M[:] = [None] * cap
            self._head = cap
            
        elif dir < 0:
            if h + n > cap:
                r = M[h:] + M[:n-cap+h]
                M[h:] = [None] * (cap-h)
                M[:n-cap+h] = [None] * (n-cap+h)
            else:
                r = M[h:h+n]
                M[h:h+n] = [None] * n
            
            self._head = (self._head + n) % cap

        else:
            t = (t - 1) % cap
            rt = cap - t
            if t - n < -1:
                r = M[-rt:-cap-1:-1] + M[-1:-n+t:-1]
                M[-rt:-cap-1:-1] = [None] * (t+1)
                M[-1:-n+t:-1] = [None] * (n-t-1)
            else:
                r = M[-rt:-rt-n:-1]
                M[-rt:-rt-n:-1] = [None] * n

        # set size
        self._size -= n
        
        return r

    def _state(self):
        return {
            'M': self._M,
            'head': self._head,
            'size': self._size
        }
        

clist = CycledList