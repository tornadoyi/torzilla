import threading as _th
from .result import Result


class ElasticPool(object):
    def __init__(self) -> None:
        self._pool = {}
        self._lock = _th.Lock()
        self._cond = _th.Condition()
        self._close = False

    def __len__(self):
        with self._cond:
            return len(self._pool)

    def apply(self, func, args=(), kwds={}):
        return self.apply_async(func, args, kwds).get()

    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        self._check_running()
        result = Result(callback, error_callback)
        t = _th.Thread(target=self.__executor__, args=(self, func, args, kwds, result))
        with self._cond:
            t.start()
            self._pool[t.native_id] = t
        return result

    def close(self):
        self._close = True

    def join(self):
        if not self._close:
            raise ValueError("Pool is still running")
        with self._cond:
            self._cond.wait_for(lambda : len(self) == 0)

    def _check_running(self):
        if self._close:
            raise ValueError("Pool not running")

    @staticmethod
    def __executor__(self, func, args, kwds, result):
        try:
            r = func(*args, **kwds)
            result._set(0, (True, r))
        except Exception as e:
            result._set(0, (False, e))
        finally:
            with self._cond:
                del self._pool[_th.get_native_id()]
                self._cond.notify_all()