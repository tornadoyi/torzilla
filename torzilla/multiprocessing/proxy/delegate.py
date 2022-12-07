from multiprocessing.managers import BaseProxy


DEFAULT_DELEGATOR_NAME = '_delegated_execute_'


class DelegateError(Exception): pass

class LocatorRuntimeError(DelegateError): pass

class LocatorNotFound(DelegateError):pass

class MethodNotFound(DelegateError): pass


def __delegate_execute__(self, locator, name, *args, **kwargs):
    try:
        obj = locator(self)
    except Exception as e:
        return LocatorRuntimeError(str(e))

    if obj is None:
        return LocatorNotFound()

    f = getattr(obj, name, None)
    if f is None:
        return MethodNotFound(name)
    
    return f(*args, **kwargs)


def delegate(tp, name=None):
    if type(tp) is not type:
        raise TypeError(f'error type for delegate, type: {tp}')
    dic = {}
    name = name or f'{str(tp)}.delegate'
    dtype = type(name, (tp,), dic)
    if hasattr(dtype, DEFAULT_DELEGATOR_NAME):
        raise TypeError(f'{tp} has been a delegator')
    setattr(dtype, DEFAULT_DELEGATOR_NAME, __delegate_execute__)
    return dtype


class Delegate(object):
    def __init__(self, proxy, locator):
        self._proxy = proxy
        self._locator = locator

    def __enter__(self):
        return self._call_delegated_method(
            '__enter__'
        )

    def __exit__(self, exc, val, tb):
        return self._call_delegated_method(
            '__exit__',
            exc, val, tb
        )
        
    def __getattr__(self, name):
        def _remote(*args, **kwargs):
            return self._call_delegated_method(
                name,
                *args, **kwargs
            )
        return _remote

    def _call_delegated_method(self, name, *args, **kwargs):
        return self._proxy._call_delegated_method(
            self._locator,
            name,
            *args, **kwargs
        )


class DelegateProxy(BaseProxy):
    def _call_delegated_method(self, locator, fname, *args, **kwargs):
        ans = self._callmethod(
            DEFAULT_DELEGATOR_NAME,  
            (locator, fname) + args, 
            kwargs
        )
        if isinstance(ans, DelegateError):
            raise ans
        return ans


def MakeProxyType(name, exposed, _cache={}):
    exposed = tuple(exposed)
    if DEFAULT_DELEGATOR_NAME not in exposed:
        exposed = exposed + (DEFAULT_DELEGATOR_NAME, )

    try:
        return _cache[(name, exposed)]
    except KeyError:
        pass

    dic = {}

    for meth in exposed:
        exec('''def %s(self, /, *args, **kwds):
        return self._callmethod(%r, args, kwds)''' % (meth, meth), dic)

    ProxyType = type(name, (DelegateProxy,), dic)
    ProxyType._exposed_ = exposed
    _cache[(name, exposed)] = ProxyType
    return ProxyType