import uuid
from collections import namedtuple

class dictedlist_iter(object):
    def __repr__(self):
        return ', '.join([str(x) for x in self])

class dictedlist_keys(dictedlist_iter):
    def __init__(self, dlist):
        self._dlist = dlist

    def __iter__(self):
        N = self._dlist._N
        cur = self._dlist._head.value
        while cur:
            yield cur
            cur = N[cur].next


class dictedlist_values(dictedlist_iter):
    def __init__(self, dlist):
        self._dlist = dlist

    def __iter__(self):
        M, N = self._dlist._M, self._dlist._N
        cur = self._dlist._head.value
        while cur:
            yield M[cur]
            cur = N[cur].next


class dictedlist_items(dictedlist_iter):
    def __init__(self, dlist):
        self._dlist = dlist

    def __iter__(self):
        M, N = self._dlist._M, self._dlist._N
        cur = self._dlist._head.value
        while cur:
            yield (cur, M[cur])
            cur = N[cur].next


class _Node(namedtuple('Node', ['prev', 'next'])):
    def copy(self, **kwargs):
        return _Node(
            kwargs.get('prev', self.prev),
            kwargs.get('next', self.next)
        )


class DictedList(object):
    def __init__(self, manager):
        self._M = manager.dict()
        self._N = manager.dict()
        self._head = manager.Value('c', None)
        self._tail = manager.Value('c', None)
    
    def __len__(self):
        return len(self._N)

    def __getitem__(self, key):
        return self._M[key]

    def __iter__(self):
        cur = self._head.value
        while cur:
            yield cur
            cur = self._N[cur].next

    def keys(self):
        return dictedlist_keys(self)

    def value(self):
        return dictedlist_values(self)

    def items(self):
        return dictedlist_items(self)
        
    def append(self, data, key=None):
        keys = None if key is None else [key]
        return self.extend([data], keys)[0]

    def appendleft(self, data, key=None):
        keys = None if key is None else [key]
        return self.extendleft([data], keys)[0]

    def clear(self):
        self._head.value = None
        self._tail.value = None
        self._M.clear()
        self._N.clear()

    def extend(self, datas, keys=None):
        keys = self._inserts(self._tail.value, datas, keys, dir=1)
        if len(keys) == 0: return []
        self._tail.value = keys[-1]
        if not self._head.value:
            self._head.value = keys[0]
        return keys

    def extendleft(self, datas, keys=None):
        keys = self._inserts(self._head.value, datas, keys, dir=-1)
        if len(keys) == 0: return []
        self._head.value = keys[-1]
        if not self._tail.value:
            self._tail.value = keys[0]
        return keys

    def insert(self, idx_key, data, key=None):
        if idx_key is None:
            raise KeyError('index key is None')
        keys = None if key is None else [key]
        key = self._inserts(idx_key, [data], keys, dir=-1)[0]
        if idx_key == self._head.value:
            self._head.value = key
        if not self._tail.value:
            self._tail.value = key

    def pop(self, size=1):
        cur_size = len(self)
        if size > cur_size:
            raise ValueError(f'exceed current size {cur_size}, size: {size}')
        
        tail = self._tail.value
        for _ in range(size):
            node = self._N[tail]
            del self._M[tail]
            del self._N[tail]
            tail = node.prev

        if tail:
            node = self._N[tail]
            self._N[tail] = node.copy(next=None)
        self._tail.value = tail

        if len(self) == 0:
            self._head.value = None
            
    def popleft(self, size=1):
        cur_size = len(self)
        if size > cur_size:
            raise ValueError(f'exceed current size {cur_size}, size: {size}')

        head = self._head.value
        for _ in range(size):
            node = self._N[head]
            del self._M[head]
            del self._N[head]
            head = node.next

        if head:
            node = self._N[head]
            self._N[head] = node.copy(prev=None)
        self._head.value = head

        if len(self) == 0:
            self._tail.value = None

    def remove(self, key):
        if key not in self._N:
            raise KeyError(f'key {key} does not exist')
            
        node = self._N[key]
        del self._M[key]
        del self._N[key]
        if self._head.value == key:
            self._head.value = node.next
        if self._tail.value == key:
            self._tail.value = node.prev
        
    def _inserts(self, root, datas, keys, dir):
        # check
        if keys is None:
            keys = [
                str(uuid.uuid1())
                for _ in range(len(datas))
            ]

        if len(keys) != len(datas):
            raise KeyError(f'key {root} does not exist')

        if root and root not in self._N:
            raise ValueError(
                f'length mismatch between datas and keys, {len(datas)} != {len(keys)}'
            )

        if len(datas) == 0: 
            return []
        
        for key in keys:
            if key not in self._N: continue
            raise KeyError(f'key {key} already exisits')

        # update root
        st, ed, step = (len(keys)-1, -1, -1) if dir < 0 else (0, len(keys), 1)
        if root:
            node = self._N[root]
            self._N[root] = node.copy(prev=keys[st])

        # insert
        _root = root
        for i in range(st, ed, step):
            key, data = keys[i], datas[i]
            self._M[key] = data

            if dir < 0:
                prev = keys[i+step] if i+step >= 0 else None
                next = _root
            else:
                prev = _root
                next = keys[i+step] if i+step < len(keys) else None
            
            self._N[key] = _Node(prev, next)
            _root = key

        return keys
        

dlist = DictedList