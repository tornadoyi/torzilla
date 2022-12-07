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
    'ready', 'successful', 'wait', 'get', '_set'
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

    def apply_async(self, *args, **kwargs):
        result = mp.current_target().manager().MultiResult(self._connections)
        self._callmethod('_apply_async', (result, args, kwargs))
        return result

    def apply(self, *args, **kwargs):
        return self.apply_async(*args, **kwargs).get()

    def connect(self, listener, processes=None):
        return threading.Gear._listen(self, listener, processes)


