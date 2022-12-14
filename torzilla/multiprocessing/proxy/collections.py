from .delegate import MakeProxyType


class CycledListProxy (MakeProxyType('CycledListProxy', (
    '__add__', '__contains__', '__copy__', '__getitem__', '__iadd__', '__imul__', '__len__',
    '__repr__', '__mul__',  '__rmul__', '__setitem__',
    'append', 'appendleft', 'capacity', 'clear', 'copy', 'count', 'extend', 'extendleft', 'index', 
    'insert', 'pop', 'popleft', 'popn', 'popnleft', 'reverse', 'sort', '_state'
    # '__iter__', '__reversed__'
))):
    pass