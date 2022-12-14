from torzilla import threading
from torzilla import multiprocessing as mp
from .delegate import MakeProxyType, Delegate


class RWLockProxy (MakeProxyType('RWLockProxy', (
    'acquire_read', 'acquire_write', 'release_read', 'release_write'
))):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rlock = Delegate(self, self._rlock_locator)
        self._wlock = Delegate(self, self._wlock_locator)

    def rlock(self):
        return self._rlock

    def wlock(self):
        return self._wlock

    def _rlock_locator(_, self):
        return self.rlock()

    def _wlock_locator(_, self):
        return self.wlock()


class ResultProxy (MakeProxyType('Result', (
    'ready', 'successful', 'wait', 'get', '_set', '__len__'
))):
    pass


class MultiResultProxy(ResultProxy):
    pass


class GearProxy (MakeProxyType('GearProxy', (
    '_apply_async', 'close', 'connections', 'join', 'running',
    '_connect_to'
))):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connections = self._callmethod('connections')

    def apply_async(self, method, args=(), kwds={}, to=None):
        slots = to or tuple(range(self._connections))
        result = mp.current_target().manager().MultiResult(len(slots))
        self._callmethod('_apply_async', (slots, (method,) + args, kwds, result))
        return result

    def apply(self, method, args=(), kwds={}, to=None):
        return self.apply_async(method, args, kwds, to).get()

    def connect(self, listener, slot=None, processes=None):
        return threading.Gear._listen(self, listener, slot, processes)


