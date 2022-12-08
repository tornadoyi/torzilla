from torzilla import NotExist
from .base import BaseParameterBuffer


class _Item(object):
    def __init__(self, value, meta):
        self.value = value
        self.meta = meta


class DictParameterBuffer(BaseParameterBuffer):
    DEFAULT_KEY = '__DEFAULT__'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.is_master():
            self._store = self.manager_create('dict')
            self._lock = self.manager_create('RWLock')
        else:
            self._store = self._master._store
            self._lock = self._master._lock

    def __len__(self):
        with self._lock.rlock():
            return len(self._store)

    def clear(self):
        with self._lock.wlock():
            self._store.clear()
    
    def get(self, value=True, meta=False, key=DEFAULT_KEY):
        with self._lock.rlock():
            item = self._store.get(key, None)
        if not item:
            raise KeyError(f'key {key} not exists')
        ans = []
        if value: ans.append(item.value)
        if meta: ans.append(item.meta)
        if len(ans) == 0:
            return None
        elif len(ans) == 1:
            return ans[0]
        else:
            return tuple(ans)

    def keys(self):
        with self._lock.rlock():
            keys = self._store.keys()
        return tuple(keys)

    def put(self, value=NotExist, meta=NotExist, key=DEFAULT_KEY):
        with self._lock.wlock():
            item = self._store.get(key, None)
            if item is None:
                if value is None:
                    raise ValueError(f'put a new key "{key}" without value')
                self._store[key] =_Item(value, meta)
            else:
                if value is not NotExist:
                    item.value = value
                if meta is not NotExist:
                    item.meta = meta

    def remove(self, key, ignore_errors=False):
        with self._lock.wlock():
            try:
                del self._store[key]
            except Exception as e:
                if not ignore_errors:
                    raise e 

    def removes(self, keys, ignore_errors=False):
        with self._lock.wlock():
            for key in keys:
                try:
                    del self._store[key]
                except Exception as e:
                    if not ignore_errors:
                        raise e 