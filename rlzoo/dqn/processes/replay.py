from torzilla.rl.replay_buffer import ListReplayBuffer
from rlzoo.zoo.role import Role


class ReplayBuffer(Role):
    def _start(self):
        master = self.manager().replay_buffer
        self._buffer = ListReplayBuffer(master=master)

    def size(self):
        return len(self._buffer)

    def append(self, *args, **kwargs):
        return self._buffer.append(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self._buffer.extend(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._buffer.sample(*args, **kwargs)

    def clear(self, *args, **kwargs):
        return self._buffer.clear(*args, **kwargs)

