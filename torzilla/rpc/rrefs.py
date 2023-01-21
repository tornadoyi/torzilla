from functools import partial
from torch.distributed import rpc as _rpc
from torch import futures
from torch._C._distributed_rpc import PyRRef


class PyRRefs(object):
    def __init__(self, rrefs) -> None:
        if not isinstance(rrefs, (list, tuple)):
            raise TypeError(f'type of rrefs is {type(rrefs)}, expected: list or tuple')
        
        for rref in rrefs:
            if isinstance(rref, PyRRef): continue
            raise TypeError(f'type of rref is {type(rref)}, expected: f{PyRRef}')

        self._rrefs = list(rrefs)

    def __add__(self, other):
        return PyRRefs(self._rrefs + other._rrefs)

    def __len__(self):
        return len(self._rrefs)

    def __getitem__(self, key):
        return self._rrefs[key]

    def __iter__(self):
        return iter(self._rrefs)

    def rpc_sync(self, *args, **kwargs):
        return RRefsProxy([
            rref.rpc_sync(*args, **kwargs) 
            for rref in self._rrefs]
        )

    def rpc_async(self, *args, **kwargs):
        return RRefsProxy([
            rref.rpc_async(*args, **kwargs) 
            for rref in self._rrefs]
        )


class RRefsProxy(object):
    def __init__(self, proxies) -> None:
        self._proxies = proxies

    @property
    def rpc_timeout(self):
        return self._proxies[0].rpc_timeout

    @property
    def rpc_api(self):
        return self._proxies[0].rpc_api

    def __getattr__(self, func_name):
        invokers = [
            getattr(proxy, func_name)
            for proxy in self._proxies
        ]
        return partial(_invoke_rpc, self.rpc_api, invokers)


def _invoke_rpc(rpc_api, invokers, *args, **kwargs):
    def _async_done(fut, fut_collect):
        try:
            fut.set_result([f.value() for f in fut_collect.value()])
        except Exception as e:
            fut.set_exception(e)

    rets = [invoker(*args, **kwargs) for invoker in invokers]
    if rpc_api == _rpc.rpc_async:
        fut = futures.Future()
        futures.collect_all(rets).add_done_callback(partial(_async_done, fut))
        return fut
    else:
        return rets
