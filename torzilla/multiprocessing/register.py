from torzilla.collections import CycledList, clist
from torzilla.threading import RWLock, Result, MultiResult, Gear
from .manager import Manager
from .proxy import *


# collecions
Manager.register('CycledList', CycledList, CycledListProxy)
Manager.register('clist', clist, CycledListProxy)


# threading
Manager.register('RWLock', delegate(RWLock), RWLockProxy)
Manager.register('Result', delegate(Result), ResultProxy)
Manager.register('MultiResult', delegate(MultiResult), MultiResultProxy)
Manager.register('Gear', delegate(Gear), GearProxy)

