import inspect
from torch.distributed import rpc as _rpc

from torzilla.core.error import *
from torzilla.core.types import NotExist
from torzilla.core import utility as U


class RRef(object):
    def __init__(self, obj) -> None:
        __verify_class(obj)
        self._tp = type(obj)
        self._rref = _rpc.RRef(obj)

    @property
    def torch_rref(self): return self._rref

    @property
    def type(self): return self._tp

    def owner(self): return self._rref.owner()

    def local_value(self): return self._rref.local_value()

    def is_owner(self): return self._rref.is_owner()

    def owner_name(self): return self._rref.owner_name()

    def __getattr__(self, name):
        U.assert_type('name', name, str)
        if self.is_owner():
            obj = self._rref.local_value()
            return getattr(obj, name)
        else:
            owner = self._rref.owner()
            f = getattr(self._tp, name, NotExist)
            if f is NotExist: raise NotExistError(name)
            return lambda *args, **kwargs: __remote_call(owner, f, *args, **kwargs)

    def sync_call(self, name, *args, **kwargs):
        pass
            

def __local_call(rref, method, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def __remote_call(rref, method, *args, **kwargs):
    args = [rref, method] + list(args)
    return _rpc.rpc_sync(rref.owner(), __local_call, args=args, kwargs=kwargs)


__RREF_MEMBERS = [t[0] for t in inspect.getmembers(RRef) if not t[0].startswith('__')]
def __verify_class(obj):
    for m in __RREF_MEMBERS:
        if not hasattr(obj, m): continue
        raise Exception(f'{obj}\'s member {m} should be mask by RRef')