
import threading as _th

class Result(object):
    def __init__(self, callback=None, error_callback=None):
        self._event = _th.Event()
        self._callback = callback
        self._error_callback = error_callback
        self._success = None
        self._value = None

    def ready(self):
        return self._event.is_set()

    def successful(self):
        if not self.ready():
            raise ValueError("{0!r} not ready".format(self))
        return self._success

    def wait(self, timeout=None):
        self._event.wait(timeout)

    def get(self, timeout=None):
        self.wait(timeout)
        if not self.ready():
            raise TimeoutError
        if self._success:
            return self._value
        else:
            raise self._value

    def _set(self, i, obj):
        if self.ready():
            raise ValueError(f'value has been set')
        self._success, self._value = obj
        if self._callback and self._success:
            self._callback(self._value)
        if self._error_callback and not self._success:
            self._error_callback(self._value)
        self._event.set()


class MultiResult(Result):
    def __init__(self, num_result, callback=None, error_callback=None):
        super().__init__(callback, error_callback)
        self._cache = [None] * num_result
        self._finishes = 0
        self._mutex = _th.Lock()

    def _set(self, i, obj):
        if self._cache[i] is not None:
            raise IndexError(f'result at {i} has been set')
        
        with self._mutex:
            if self._cache[i] is not None:
                raise IndexError(f'value of index {i} has been set')
            self._cache[i] = obj
            self._finishes += 1

        if self._finishes == len(self._cache):
            success, value = True, []
            for (suc, val) in self._cache:
                if suc: 
                    value.append(val)
                    continue
                success, value = False, val
                break
            super()._set(i, (success, value))