from .error import *
from .types import *
from . import utility as U

class _Container(dict): pass

class PathStore(object):
    def __init__(self) -> None:
        self._d = {}

    def get(self, path, default=NotExist):
        U.assert_type('path', path, str)
        s = path.split('/')
        return self._find(path, s, 0, len(s), 0)

    def set(self, path, value):
        U.assert_type('path', path, str)
        s = path.split('/')
        root = self._find(path, s, 0, len(s)-1, 1)
        if not isinstance(root, _Container):
            raise InvalidArgumentError(
                'path', 
                path, 
                desc=f'Type of {"/".join(s[:-1])} is {type(root)}, expected container'
            )
        root[s[-1]] = value
        
    def remove(self, path):
        U.assert_type('path', path, str)
        s = path.split('/')
        root = self._find(path, s, 0, len(s)-1, 1)
        if not isinstance(root, _Container):
            raise InvalidArgumentError(
                'path', 
                path, 
                desc=f'Type of {"/".join(s[:-1])} is {type(root)}, expected container'
            )
        del root[s[-1]]


    def _find(self, paths, len, empty_action):   # 0: error  1: new
        root = self._d
        for i in range(0, len, 1):
            k = paths[i]
            if not isinstance(root, _Container):
                raise InvalidArgumentError(
                    'path', 
                    '/'.join(paths), 
                    desc=f'Type of {"/".join(paths[:i+1])} is {type(root)}, expected container'
                )

            v = root.get(k, NotExist)
            if v is NotExist:
                if empty_action == 1:
                    v = root[k] =_Container()
                else:
                    raise NotExistError(
                        '/'.join(paths), 
                        desc=f'{"/".join(paths[:i+1])} is not exist'
                    )
            root = v
        return root