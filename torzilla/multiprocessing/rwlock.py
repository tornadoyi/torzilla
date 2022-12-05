from torzilla.threading import _Core, _RWLock
import multiprocessing as mp

class _ProcessCore(_Core):
    def __init__(self, w_first):
        super().__init__(w_first)
        self.__cond = mp.Condition()
        self.__state = mp.Value('l', 0)  
        self.__n_w_wait = mp.Value('l', 0)

    @property
    def _n_w_wait(self): return self.__n_w_wait.value

    @_n_w_wait.setter
    def _n_w_wait(self, v): self.__n_w_wait.value = v

    @property
    def _cond(self): return self.__cond
    
    @property
    def _state(self): return self.__state.value

    @_state.setter
    def _state(self, v): self.__state.value = v


class RWLock(_RWLock):
    __CORE__ = _ProcessCore

    