from torzilla.rl.replay_buffer import ListReplayBuffer
from rlzoo.zoo.role import Role


class ReplayBuffer(Role):
    def _on_start(self):
        master = self.manager.replay_buffer
        self._buffer = ListReplayBuffer(master=master)
        self._buffer.start()

    def _on_exit(self, *args, **kwargs):
        self._buffer.exit()

    def __getattr__(self, name):
        return getattr(self._buffer, name)

    def size(self):
        return len(self._buffer)

    def put(self, *args, **kwargs):
        return self._buffer.put(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._buffer.sample(*args, **kwargs)

    def clear(self, *args, **kwargs):
        return self._buffer.clear(*args, **kwargs)

