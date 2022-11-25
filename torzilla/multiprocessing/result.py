from torch import multiprocessing as mp
from multiprocessing.pool import ApplyResult

class Result(ApplyResult):
    def __init__(self, manager, callback=None, error_callback=None):
        self._event = manager.Event()
        self._callback = callback
        self._error_callback = error_callback
        self._result = manager.Namespace()
        self._result.success = None
        self._result.value = None

    def ready(self):
        return self._event.is_set()

    def successful(self):
        if not self.ready():
            raise ValueError("{0!r} not ready".format(self))
        return self._result.success

    def wait(self, timeout=None):
        self._event.wait(timeout)

    def get(self, timeout=None):
        self.wait(timeout)
        if not self.ready():
            raise TimeoutError
        if self._result.success:
            return self._result.value
        else:
            raise self._result.value

    def _set(self, i, obj):
        self._result.success, self._result.value = obj
        if self._callback and self._result.success:
            self._callback(self._result.value)
        if self._error_callback and not self._result.success:
            self._error_callback(self._result.value)
        self._event.set()


class MultiResult(Result):
    def __init__(self, num_result, manager, callback=None, error_callback=None):
        super().__init__(manager, callback, error_callback)
        self._cache = manager.list([None] * num_result)
        self._finishes = manager.Value('l', 0)
        self._mutex = manager.Lock()

    def _set(self, i, obj):
        if self._cache[i] is not None:
            raise IndexError(f'result at {i} has been set')
        self._cache[i] = obj
        with self._mutex:
            self._finishes.value += 1

        if self._finishes.value == len(self._cache):
            success, value = True, []
            for (suc, val) in self._cache:
                if suc: 
                    value.append(val)
                    continue
                success, value = False, val
                break
            super()._set(i, (success, value))
