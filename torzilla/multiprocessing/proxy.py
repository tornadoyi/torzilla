from multiprocessing.managers import MakeProxyType


class _CallWrapper(object):
    __FUNCTIONS__ = {}

    def __init__(self, proxy):
        self._proxy = proxy

    def __getattr__(self, name):
        call_name = self.__FUNCTIONS__.get(name, None)
        if call_name is None:
            raise Exception(f'{call_name}')
        return lambda *args, **kwargs: self._proxy._callmethod(call_name, args, kwargs)


class CycledListProxy (MakeProxyType('CycledListProxy', (
    '__add__', '__contains__', '__copy__', '__getitem__', '__iadd__', '__imul__', '__len__',
    '__repr__', '__mul__',  '__rmul__', '__setitem__',
    'append', 'appendleft', 'capacity', 'clear', 'copy', 'count', 'extend', 'extendleft', 'index', 
    'insert', 'pop', 'popleft', 'popn', 'popnleft', 'reverse', 'sort', '_state'
    # '__iter__', '__reversed__'
))):
    pass


class _LockWrapper(_CallWrapper):
    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc, val, tb):
        return self.release()

class _RLockWrapper(_LockWrapper):
    __FUNCTIONS__ = {
        'acquire': 'acquire_read',
        'release': 'release_read',
    }

class _WLockWrapper(_LockWrapper):
    __FUNCTIONS__ = {
        'acquire': 'acquire_write',
        'release': 'release_write',
    }


class RWLockProxy (MakeProxyType('RWLockProxy', (
    'acquire_read', 'acquire_write', 'release_read', 'release_write'
))):
    def rlock(self):
        return _RLockWrapper(self)

    def wlock(self):
        return _WLockWrapper(self)


