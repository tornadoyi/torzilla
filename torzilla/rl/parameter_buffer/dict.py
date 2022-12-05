from collections import namedtuple
from torzilla import multiprocessing as mp
from .base import BaseParameterBuffer

__Node = namedtuple('_Node', ['prev', 'next'], defaults=[None, None])
class _Node(__Node):
    def copy(self, **kwargs):
        return _Node(
            kwargs.get('prev', self.prev),
            kwargs.get('next', self.next),
        )
        
class DictParameterBuffer(BaseParameterBuffer):
    DEFAULT_KEY = '__DEFAULT__'

    def __init__(self, capacity=None, **kwargs):
        super().__init__(**kwargs)
        if self.is_master():
            self._capacity = capacity
        else:
            self._capacity = self.master.capacity

    @property
    def capacity(self): return self._capacity

    def __len__(self):
        with self._lock.rlock():
            return len(self._M)

    def size(self):
        return len(self)

    def _on_start(self):
        super()._on_start()
        if self.is_master():
            manager = mp.Process.current().manager
            self._M = manager.dict()
            self._N = manager.dict()
            self._attr = manager.Namespace()
            self._attr.head = None
            self._attr.tail = None
            self._lock = manager.RWLock()
        else:
            self._M = self.master._M
            self._N = self.master._N
            self._attr = self.master._attr
            self._lock = self.master._lock

    def put(self, data, meta=None, key=DEFAULT_KEY):
        with self._lock.wlock():
            # update data
            self._M[key] = (data, meta)

            # update node
            node = self._N.get(key)
            if node is None:
                node = self._N[key] = _Node()
            self._refresh_node(node)

    def get(self, param=True, meta=False, key=DEFAULT_KEY):
        with self._lock.rlock():
            pass

    def clear(self):
        with self._lock.wlock():
            self._M.clear()
            self._attr.head = None
            self._attr.tail = None

    def _refresh_node(self, key, node):
        # remove
        if node.prev:
            prev = self._N[node.prev]
            self._N[node.prev] = prev.copy(next=node.next)
        if node.next:
            next = self._N[node.next]
            self._N[node.next] = next.copy(prev=node.prev)

        # update tail
        node = node.copy(prev=self._attr.tail, next=None)
        if self._attr.tail:
            tail = self._N[self._attr.tail]
            self._N[self._attr.tail] = tail.copy(next=key)
        self._attr.tail = key

        # update head