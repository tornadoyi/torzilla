import threading as _th

__GL__ = _th.RLock()

def GL(): return __GL__

def synchronized(lock=None):
    if lock is None:
        global __GL__
        lock = __GL__

    def wrap(f):
        def locked_f(*args, **kw):
            with lock:
                return f(*args, **kw)
        return locked_f
    return wrap