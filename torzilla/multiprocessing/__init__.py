from torch.multiprocessing import *
from .lanucher import lanuch
from .target import Target, current_target, current_target_rref
from .manager import Manager
from .rwlock import RWLock
from .proxy.delegate import *
from . import register as _

