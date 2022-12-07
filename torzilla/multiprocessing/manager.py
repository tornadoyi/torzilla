from multiprocessing.managers import SyncManager as _SyncManager

class Manager(_SyncManager):
    def start(self, *args, **kwargs):
        ret = super().start(*args, **kwargs)
        self._start()
        return ret

    def exit(self):
        self._exit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
        super().__exit__(exc_type, exc_val, exc_tb)

    def _start(self): pass

    def _exit(self): pass