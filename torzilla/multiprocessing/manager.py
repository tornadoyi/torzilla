from multiprocessing.managers import SyncManager as _SyncManager

__MANAGER__ = None

class Manager(_SyncManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global __MANAGER__
        __MANAGER__ = self

    
    def start(self, *args, **kwargs) -> None:
        ret = super().start(*args, **kwargs)
        self._on_start(*args, **kwargs)
        return ret

    def _on_start(self, *args, **kwargs):
        pass