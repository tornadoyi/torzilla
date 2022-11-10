


class Context(object):
    def __init__(self):
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc, val, tb):
        self.exit()

    def start(self, *args, **kwargs): 
        if self._started:
            raise Exception(f'{type(self)} start repeatedly')
        self._started = True
        self._on_start(*args, **kwargs)

    def exit(self, *args, **kwargs):
        if not self._started:
            raise Exception(f'{type(self)} need to exit after start')
        self._on_exit(*args, **kwargs)
        self._started = False

    def _on_start(self, *args, **kwargs): pass

    def _on_exit(self, *args, **kwargs): pass