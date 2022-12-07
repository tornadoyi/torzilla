import sys
import traceback
from collections import namedtuple
from multiprocessing.pool import ThreadPool
import threading as _th
from .result import MultiResult

_CallInfo = namedtuple('_CallInfo', ['args', 'kwargs', 'result'])

class Gear(object):
    def __init__(self, connections):
        self._connections = connections
        self._running = True
        self._cond = _th.Condition()
        self._lock = _th.Lock()
        self._id = 0
        self._num_ready = 0
        self._call = None

    def apply_async(self, *args, **kwargs):
        result = MultiResult(self._connections)
        self._apply_async(result, *args, **kwargs)
        return result

    def apply(self, *args, **kwargs):
        return self.apply_async(args, kwargs).get()

    def close(self):
        if not self._running:
            return
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def connect(self, listener, processes=None):
        return self._listen(self, listener, processes)

    def connections(self):
        return self._connections

    def join(self):
        with self._cond:
            self._cond.wait_for(lambda : not self._running)

    def running(self):
        return self._running
        
    def _apply_async(self, result, args, kwargs):
        def _cond_ready():
            return (
                not self._running or
                self._num_ready >= self._connections
            )
        with self._lock, self._cond:
            self._cond.wait_for(_cond_ready)
            self._num_ready = 0
            self._id = (self._id + 1) % 2
            self._call = _CallInfo(
                args = args,
                kwargs = kwargs,
                result = result,
            )
            self._cond.notify_all()
            self._cond.wait_for(_cond_ready)

    def _connect_to(self, id):
        def _cond_call():
            return (
                not self._running or
                self._id != id
            )
        with self._cond:
            index = self._num_ready
            self._num_ready += 1
            self._cond.notify_all()
            self._cond.wait_for(_cond_call)
            if not self._running:
                return False, None, None, None
            return True, index, self._id, self._call

    @staticmethod
    def _listen(gear, listener, processes=None):
        pool = ThreadPool(processes=processes)

        def _call(index, args, kwargs, result):
            try:
                r = listener(*args, **kwargs)
                result._set(index, (True, r))
            except Exception as e:
                e = type(e)(traceback.format_exc(chain=False))
                result._set(index, (False, e))

        def _error_callback(e):
            print(
                ''.join(traceback.format_exception(
                    type(e), value=e, tb=e.__traceback__
                )),
                file=sys.stderr
            )

        def _loop():
            last_call_id = 0
            running = True
            while running:
                running, index, last_call_id, call = gear._connect_to(last_call_id)
                if not running: break
                pool.apply_async(
                    _call,
                    args=(index, call.args, call.kwargs, call.result),
                    error_callback=_error_callback,
                )
            pool.close()
        
        thread = _th.Thread(target=_loop)
        thread.start()
        return Connection(gear, thread, pool)

    def _create_result(self):
        raise NotImplementedError()


class Connection(object):
    def __init__(self, gear, thread, pool):
        self._gear = gear
        self._thread = thread
        self._pool = pool
    
    def join(self):
        if self._gear is None: return
        self._gear.join()
        self._pool.close()
        self._pool.join()
        self._thread.join()
        self._gear = None
        self._pool = None