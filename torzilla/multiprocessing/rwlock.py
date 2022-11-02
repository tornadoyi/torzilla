import time
import multiprocessing as mp

class _Core(object):
    def __init__(self, write_first, manager):
        mod = mp if manager is None else manager
        self._cond = mod.Condition()
        self._state = mod.Value('l', 0)  # >0: read, <0: write 0: no ops
        self._num_w_wait = mod.Value('l', 0)
        self._write_first = mod.Value('b', write_first)

    def state(self): return self._state.value

    def acquire_read(self, timeout=None):
        with self._cond:
            return self.wait_for(self._acquire_read, timeout)

    def acquire_write(self, timeout=None):
        with self._cond:
            self._num_w_wait.value += 1
            try:
                return self.wait_for(self._acquire_write, timeout)
            finally:
                self._num_w_wait.value -= 1

    def wait_for(self, predicate, timeout=None):
        wait_time = timeout
        end_time = None if wait_time is None else time.time() + wait_time
        result = predicate()
        while not result:
            if wait_time is not None:
                wait_time = end_time - time.time()
            self._cond.wait(wait_time)
            result = predicate()
        return result

    def release(self):
        with self._cond:
            if self._state.value > 0:
                self._state.value -= 1
            else:
                self._state.value += 1

            if self._state.value == 0:
                self._cond.notify_all()

    def _acquire_read(self):
        if (
            self._state.value < 0 or
            (self._write_first.value == 1 and self._num_w_wait.value > 0)
        ):
            return False

        self._state.value += 1
        return True

    def _acquire_write(self):
        if self._state.value != 0: return False
        self._state.value -= 1
        return True

    
class _Lock(object):
    def __init__(self, core):
        self._core = core

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc, val, tb):
        self.release()
    
    @staticmethod
    def _timeout(block, timeout):
        if timeout and timeout < 0:
            raise ValueError(f"Invalid timeout {timeout}")
        if not block and timeout is not None:
            raise ValueError("Can't specify a timeout when non-blocking")
        return timeout

    def release(self):
        return self._core.release()

class _RLock(_Lock):
    def acquire(self, block=True, timeout=None):
        return self._core.acquire_read(self._timeout(block, timeout))

class _WLock(_Lock):
    def acquire(self, block=True, timeout=None):
        return self._core.acquire_write(self._timeout(block, timeout))

class RWLock():
    def __init__(self, write_first=True, manager=None):
        self._core = _Core(write_first, manager)
        self._reader_lock = _RLock(self._core)
        self._writer_lock = _WLock(self._core)

    def locked(self): return self._core.state() != 0

    def rlocked(self): return self._core.state() < 0

    def wlocked(self): return self._core.state() != 0

    def __enter__(self):
        self._writer_lock.acquire()

    def __exit__(self, exc, val, tb):
        self._writer_lock.release()

    def reader_lock(self):
        return self._reader_lock

    def writer_lock(self):
        return self._writer_lock