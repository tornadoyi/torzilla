import time
import threading as _th

class _Core(object):
    def __init__(self, w_first):
        '''
        _state:
              0: no ops
             >0: read
             <0: write 
        '''
        self._w_first = w_first

    @property
    def _n_w_wait(self): raise NotImplementedError('_n_w_wait')

    @_n_w_wait.setter
    def _n_w_wait(self, v): raise NotImplementedError('_n_w_wait')

    @property
    def _cond(self): raise NotImplementedError('_cond')
    
    @property
    def _state(self): raise NotImplementedError('state')

    @_state.setter
    def _state(self, v): raise NotImplementedError('_state')
    
    def acquire_read(self, timeout=None):
        with self._cond:
            return self.wait_for(self._acquire_read, timeout)

    def acquire_write(self, timeout=None):
        with self._cond:
            self._n_w_wait = self._n_w_wait + 1
            try:
                return self.wait_for(self._acquire_write, timeout)
            finally:
                self._n_w_wait = self._n_w_wait - 1

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
            if self._state > 0:
                self._state = self._state - 1
            else:
                self._state = self._state + 1

            if self._state == 0:
                self._cond.notify_all()

    def _acquire_read(self):
        if (
            self._state < 0 or
            (self._w_first and self._n_w_wait > 0)
        ):
            return False

        self._state = self._state + 1
        return True

    def _acquire_write(self):
        if self._state != 0: return False
        self._state = self._state - 1
        return True

    def state(self):
        return self._state

    
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

class _RWLock():
    __CORE__ = None

    def __init__(self, w_first=True):
        self._core = self.__CORE__(w_first)
        self._rlock = _RLock(self._core)
        self._wlock = _WLock(self._core)

    def locked(self): return self._core.state() != 0

    def rlocked(self): return self._core.state() < 0

    def wlocked(self): return self._core.state() != 0

    def __enter__(self):
        self.acquire_write()

    def __exit__(self, exc, val, tb):
        self.release_write()

    def rlock(self):
        return self._rlock

    def wlock(self):
        return self._wlock

    def wait_for(self, predicate, timeout=None):
        return self._core.wait_for(predicate, timeout)
        
    def acquire_read(self, *args, **kwargs):
        self._rlock.acquire(*args, **kwargs)

    def acquire_write(self, *args, **kwargs):
        self._wlock.acquire(*args, **kwargs)

    def release_read(self):
        self._rlock.release()

    def release_write(self):
        self._wlock.release()


class _ThreadCore(_Core):
    def __init__(self, w_first):
        super().__init__(w_first)
        self.__cond = _th.Condition()
        self.__state = 0  
        self.__n_w_wait = 0

    @property
    def _n_w_wait(self): return self.__n_w_wait

    @_n_w_wait.setter
    def _n_w_wait(self, v): self.__n_w_wait = v

    @property
    def _cond(self): return self.__cond
    
    @property
    def _state(self): return self.__state

    @_state.setter
    def _state(self, v): self.__state = v


class RWLock(_RWLock):
    __CORE__ = _ThreadCore