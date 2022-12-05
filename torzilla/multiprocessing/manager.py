from multiprocessing.managers import SyncManager as _SyncManager
from torzilla.core import *
from torzilla.collections import *
from torzilla.threading import RWLock
from .dlist import DictedList, dlist
from .result import Result, MultiResult
from .gear import Gear
from .proxy import *


__MANAGER__ = None

class Manager(_SyncManager, object.Context):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        object.Context.__init__(self)
        self._shutdown = False

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc, val, tb):
        super().__exit__(exc, val, tb)
        object.Context.__exit__(self, exc, val, tb)

    def start(self, *args, **kwargs):
        global __MANAGER__
        if __MANAGER__ is not None:
            raise Exception('Manager must be singleton')
        __MANAGER__ = self

        ret = super().start(*args, **kwargs)
        object.Context.start(self, *args, **kwargs)
        return ret

    def exit(self):
        object.Context.exit(self)
        if not self._shutdown: 
            self.shutdown()
        global __MANAGER__
        __MANAGER__ = None

    def shutdown(self) -> None:
        self._shutdown = True
        return super().shutdown()

    def DictedList(self, *args, **kwargs):
        kwargs['manager'] = self
        return DictedList(*args, **kwargs)

    def dlist(self, *args, **kwargs):
        kwargs['manager'] = self
        return dlist(*args, **kwargs)

    def Result(self, *args, **kwargs):
        kwargs['manager'] = self
        return Result(*args, **kwargs)

    def MultiResult(self, *args, **kwargs):
        kwargs['manager'] = self
        return MultiResult(*args, **kwargs)

    def Gear(self, *args, **kwargs):
        kwargs['manager'] = self
        return Gear(*args, **kwargs)




# collecions
Manager.register('CycledList', CycledList, CycledListProxy)
Manager.register('clist', clist, CycledListProxy)


# lock
Manager.register('RWLock', RWLock, RWLockProxy)


__DEFAULT_MANAGER__ = Manager
def set_manager_type(manager_type):
    global __DEFAULT_MANAGER__
    assert_subclass(manager_type, Manager)
    __DEFAULT_MANAGER__ = manager_type

def get_manager_type(): return __DEFAULT_MANAGER__