import sys
import traceback
import numbers
from multiprocessing.pool import ThreadPool
import threading as _th
from .result import Result, MultiResult


class Gear(object):
    def __init__(self, connections):
        self._connections = connections
        self._running = True
        self._cond = _th.Condition()
        self._lock = _th.Lock()
        self._num_ready = 0
        self._slots = [None] * connections

        # call
        self._id = 0
        self._args = None
        self._kwargs = None
        self._slot2index = None
        self._result = None
    
    def apply_async(self, method, args=(), kwds={}, to=None):
        if isinstance(to, numbers.Number):
            slots = (to, )
            result = Result()
        else:
            slots = to or tuple(range(self._connections))
            result = MultiResult(len(slots))
        self._apply_async(slots, (method,) + args, kwds, result)
        return result

    def apply(self, method, args=(), kwds={}, to=None):
        return self.apply_async(method, args, kwds, to).get()

    def close(self):
        if not self._running:
            return
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def connect(self, listener, slot=None, processes=None):
        return self._listen(self, listener, slot, processes)

    def connections(self):
        return self._connections

    def join(self):
        with self._cond:
            self._cond.wait_for(lambda : not self._running)

    def running(self):
        return self._running
        
    def _apply_async(self, slots, args, kwargs, result):
        # check
        if len(slots) != len(result):
            raise ValueError(f'slots is not equal results, {len(slots)} != {len(result)}')
        for s in slots:
            if 0 <= s < self._connections: continue
            raise IndexError(f'invalid slot {s}, expected: [0, {self._connections})')

        def _cond_ready():
            return (
                not self._running or
                self._num_ready >= self._connections
            )
        with self._lock, self._cond:
            self._cond.wait_for(_cond_ready)
            for s in slots:
                self._slots[s] = None
            self._num_ready -= len(slots)
            self._id = (self._id + 1) % 2
            self._args = args
            self._kwargs = kwargs
            self._result = result
            self._slot2index = dict([(i, s) for i, s in enumerate(slots)])
            self._cond.notify_all()
            self._cond.wait_for(_cond_ready)

    def _connect_to(self, slot, id):
        def _cond_call():
            return (
                not self._running or
                (self._id != id and slot in self._slot2index)
            )
        with self._cond:
            if slot is None:
                slot = self._num_ready
            if self._slots[slot] is not None:
                raise IndexError(f'slot {slot} has been occupied')
            self._slots[slot] = True
            self._num_ready += 1
            self._cond.notify_all()
            self._cond.wait_for(_cond_call)
            if not self._running:
                return False, None
            index = self._slot2index.get(slot, None)
            return True, (self._id, index, self._args, self._kwargs, self._result)

    @staticmethod
    def _listen(gear, listener, slot, processes):

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

        def _loop(pool):
            last_id = 0
            running = True
            while running:
                running, call_info = gear._connect_to(slot, last_id)
                if not running: break
                last_id, index, args, kwargs, result = call_info
                if index is None: continue
                pool.apply_async(
                    _call, 
                    args = (index, args, kwargs, result),
                    error_callback = _error_callback
                )
        
        pool = ThreadPool(processes)
        thread = _th.Thread(target=_loop, args=(pool, ))
        thread.start()
        return Connection(gear, thread, pool)


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