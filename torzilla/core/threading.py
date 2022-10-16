from threading import *

__GL__ = RLock()

def GL(): return __GL__

def synchronized():
    def wrap(f):
        def locked_f(*args, **kw):
            global __GL__
            with __GL__:
                return f(*args, **kw)
        return locked_f
    return wrap