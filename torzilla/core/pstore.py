from enum import Enum
from .types import *
from .assertion import *
from torzilla import threading

class _Container(dict): pass

class _Action(Enum):
    Error = 0
    Create = 1

class NotExistError(Exception): pass

class _BaseStore(object):
    def __init__(self) -> None:
        self._d = {}

    def __getitem__(self, path):
        return self.get(path)

    def __setitem__(self, path, value):
        return self.set(path, value)

    def _get(self, paths, default=NotExist):
        if default is NotExist:
            return self._find(paths, 0, len(paths), _Action.Error)
        else:
            try:
                return self._find(paths, 0, len(paths), _Action.Error)
            except NotExistError:
                return default

    def set(self, paths, value):
        root = self._find(paths, 0, len(paths)-1, 1)
        if not isinstance(root, _Container):
            raise KeyError(f'path {"/".join(paths[:-1])} is {type(root)}, expected container')
        root[paths[-1]] = value
        
    def _remove(self, paths):
        root = self._find(paths, 0, len(paths)-1, 1)
        if not isinstance(root, _Container):
            raise KeyError(f'path {"/".join(paths[:-1])} is {type(root)}, expected container')
        del root[paths[-1]]

    def _find(self, paths, len, action):   # 0: error  1: new
        root = self._d
        for i in range(0, len, 1):
            k = paths[i]
            if not isinstance(root, _Container):
                raise KeyError(f'path {"/".join(paths[:i+1])} is {type(root)}, expected container')

            v = root.get(k, NotExist)
            if v is NotExist:
                if action == _Action.Create:
                    v = root[k] =_Container()
                else:
                    raise NotExistError(f'path {"/".join(paths[:i+1])} is not exist')
            root = v
        return root

    def _split_path(path):
        assert_type(path, str)
        return path.split('/')

class PathStore(_BaseStore):
    def get(self, path, default=NotExist):
        return self._get(self._split_path(path), default)

    def set(self, path, value):
        return self._set(self._split_path(path), value)
        
    def remove(self, path):
        return self._remove(self._split_path(path))

class LockedPathStore(_BaseStore):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
    
    def get(self, path, default=NotExist):
        with self._lock:
            return self._get(self._split_path(path), default)

    def set(self, path, value):
        with self._lock:
            return self._set(self._split_path(path), value)
        
    def remove(self, path):
        with self._lock:
            return self._remove(self._split_path(path))