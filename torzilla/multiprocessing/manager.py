from multiprocessing.managers import SyncManager as _SyncManager
from .rwlock import RWLock
from torzilla.core import utility as U

__MANAGER__ = None

class Manager(_SyncManager):

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc, val, tb):
        super().__exit__(exc, val, tb)
        self.exit()

    def start(self, *args, **kwargs):
        global __MANAGER__
        if __MANAGER__ is not None:
            raise Exception('Manager must be singleton')
        __MANAGER__ = self

        ret = super().start(*args, **kwargs)
        self._on_start(*args, **kwargs)
        return ret

    def exit(self):
        self._on_exit()
        global __MANAGER__
        __MANAGER__ = None

    def _on_start(self, *args, **kwargs): pass

    def _on_exit(self): pass

    def RWLock(self, *args, **kwargs):
        kwargs['manager'] = self
        return RWLock(*args, **kwargs)


class SharedManager(Manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_data = None
        self._shared_lock = None

    @property
    def shared_data(self): return self._shared_data

    @property
    def shared_lock(self): return self._shared_lock

    def start(self, *args, **kwargs):
        ret = super().start(*args, **kwargs)
        self._shared_data = self.dict()
        self._shared_lock = self.RLock()
        return ret

__DEFAULT_MANAGER__ = Manager
def set_manager_type(manager_type):
    global __DEFAULT_MANAGER__
    U.assert_subclass(manager_type, Manager)
    __DEFAULT_MANAGER__ = manager_type

def get_manager_type(): return __DEFAULT_MANAGER__