import traceback
from multiprocessing.pool import ThreadPool

class Gear(object):
    def __init__(self, connections, manager):
        self._manager = manager
        self._connections = connections
        self._running = manager.Value('b', True)
        self._cond = manager.Condition()
        self._lock = manager.Lock()
        self._num_wait = manager.Value('l', 0)

        self._call = manager.Namespace()
        self._call.args = None
        self._call.kwargs = None
        self._call.result = None
        self._call.id = 0

    def apply_async(self, *args, **kwargs):
        def _cond_ready():
            return (
                not self._running.value or
                self._num_wait.value >= self._connections
            )
                
        with self._lock, self._cond:
            self._cond.wait_for(_cond_ready)
            self._num_wait.value = 0   
            self._call.args = args
            self._call.kwargs = kwargs
            self._call.result = self._manager.MultiResult(self._connections)
            self._call.id = (self._call.id + 1) % 2
            self._cond.notify_all()
            self._cond.wait_for(_cond_ready)
            return self._call.result

    def apply(self, *args, **kwargs):
        ar = self.apply_async(*args, **kwargs)
        r = ar.get()
        return r

    def connect(self, func, processes=None):
        pool = ThreadPool(processes=processes)
        conn_index = None
        last_apply_id = 0
        
        def _cond_call():
            return (
                not self._running.value or 
                self._call.id != last_apply_id
            )

        def _call_func(index, result, *args, **kwargs):
            try:
                r = func(*args, **kwargs)
                result._set(index, (True, r))
            except Exception as e:
                e = type(e)(traceback.format_exc())
                result._set(index, (False, e))

        def _listen():
            nonlocal conn_index
            nonlocal last_apply_id

            # wait
            if conn_index is None:
                conn_index = self._num_wait.value
            self._num_wait.value += 1
            self._cond.notify_all()
            self._cond.wait_for(_cond_call)
            if not self._running.value: 
                return

            # call
            pool.apply_async(
                _call_func,
                args=[conn_index, self._call.result] + list(self._call.args),
                kwds=self._call.kwargs
            )
            last_apply_id = self._call.id

        def _loop():
            while self._running.value:
                with self._cond:
                    _listen()
            pool.close()
        
        pool.apply_async(_loop)
        return Connection(self, pool)

    def close(self):
        if not self._running.value:
            return
        with self._cond:
            self._running.value = False
            self._cond.notify_all()


class Connection(object):
    def __init__(self, gear, pool):
        self._gear = gear
        self._pool = pool
    
    def join(self):
        if self._gear is None: return
        with self._gear._cond:
            self._gear._cond.wait_for(self._cond_close)
        self._pool.close()
        self._pool.join()
        self._gear = None
        self._pool = None

    def _cond_close(self):
        return not self._gear._running.value