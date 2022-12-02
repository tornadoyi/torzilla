from operator import itemgetter


class CycledList(object):
    def __init__(self, capacity, manager):
        self._M = manager.list([None] * capacity)
        self._head = manager.Value('l', -1)
        self._tail = manager.Value('l', 0)
    
    def __len__(self):
        capacity, h, t = len(self._M), self._head.value, self._tail.value
        if h < 0: 
            return 0
        elif h < t:
            return t - h
        elif h > t:
            return capacity - h + t
        else:
            return capacity

    def __iter__(self):
        size = len(self)
        for i in range(size):
            yield self[i]

    def __iadd__(self, other):
        return self.extend(other)

    def __imul__(self, size):
        return self.extend(self[:] * size)

    def __getitem__(self, key):
        M, h, = self._M, self._head.value
        size, capacity = len(self), len(M)

        def _index(index):
            if index < 0 or index >= size:
                raise IndexError(f'list index out of range, index: {index} range: (0, {size-1})')
            return M[(h + index) % capacity]

        def _list(indexes):
            for index in indexes:
                if index < 0 or index >= size:
                    raise IndexError(f'list index out of range, index: {index} range: (0, {size-1})')
            getter = itemgetter(*((h + index) % capacity for index in indexes))
            return getter(M)

        def _slice(start, stop, step):
            if stop <= start: return []
            st, ed = (h + start) % capacity, (h + stop) % capacity
            if ed <= st:
                if (capacity - st - 1) % step == 0:
                    return M[st:capacity:step] + M[1:ed:step]
                else:
                    return M[st:capacity:step] + M[0:ed:step]
            else:
                return M[st:ed:step]

        # check
        if isinstance(key, int):
            index = size + key if key < 0 else key
            return _index(index)

        elif isinstance(key, list):
            return _list(key)

        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or size
            step = key.step or 1
            if step <= 0:
                raise ValueError('slice step cannot less equal zero')

            # correct
            if start < 0: start = max(start + size, 0)
            if stop < 0: stop = max(stop + size, 0)
            start = min(start, size)
            stop = min(stop, size)
            return _slice(start, stop, step)

        else:
            raise TypeError(f'invalid key type {type(key)}, expected: int, slice, list')

    def capacity(self):
        return len(self._M)

    def full(self):
        return len(self) == self.capacity(self)

    def append(self, data):
        return self.extend([data])

    def extend(self, datas):
        # check
        raw_data_size = len(datas)
        if raw_data_size == 0: return

        M, head, tail = self._M, self._head, self._tail
        capacity = len(M)

        # truncate data to retain only last number equals to capacity
        offset = 0
        if raw_data_size > capacity:
            datas = datas[-capacity:]
            offset = raw_data_size % capacity
        data_size = len(datas)

        _h, _t = head.value, tail.value
        t = _t + offset
        if t + data_size > capacity:
            M[t:capacity] = datas[:capacity-t]
            M[0:data_size-capacity+t] = datas[capacity-t:]
        else:
            M[t:t+data_size] = datas[:]
        
        tail.value = (t + data_size) % capacity        
        
        if _h != _t and _t + raw_data_size < capacity:
            head.value = 0
        else:
            head.value = tail.value

    def pop(self, size=1):
        cur_size = len(self)
        if size > cur_size:
            raise ValueError(f'exceed current size {cur_size}, size: {size}')
        
        M, head, tail = self._M, self._head, self._tail
        capacity, h, t = len(M), head.value, tail.value
        if size > t:
            t = capacity - size + t
        else:
            t -= size
        if h == t:
            head.value = -1
            tail.value = 0
        else:
            tail.value = t
            
    def popleft(self, size=1):
        cur_size = len(self)
        if size > cur_size:
            raise ValueError(f'exceed current size {cur_size}, size: {size}')

        M, head, tail = self._M, self._head, self._tail
        capacity, h, t = len(M), head.value, tail.value
        if size > capacity - h:
            h = size - capacity + h
        else:
            h += size
        if h == t:
            head.value = -1
            tail.value = 0
        else:
            head.value = h

    def clear(self):
        self._head.value = -1
        self._tail.value = 0

clist = CycledList