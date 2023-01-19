from torzilla.rl.parameter_buffer import DictParameterBuffer
from rlzoo.zoo.role import Role


class ParameterServer(Role):
    def _start(self):
        master = self.manager().ps
        self._buffer = DictParameterBuffer(master=master)

    def size(self):
        return len(self._buffer)

    def get(self, *args, **kwargs):
        return self._buffer.get(*args, **kwargs)

    def keys(self):
        return self._buffer.keys()

    def put(self, *args, **kwargs):
        return self._buffer.put(*args, **kwargs)