from torzilla import multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer

class Manager(mp.Manager):
    def _on_start(self, *args, **kwargs):
        config = self.kwargs['config']

        # replay buffer
        cfg = config['replay_buffer']
        self.replay_buffer = ListReplayBuffer(
            capacity=cfg['capacity'], 
            max_cache_size=cfg['max_cache_size']
        )

        # worker
        self.worker = self.manager.dict(
            cond = self.manager.Condition(),
            
        )


class Worker(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        pass


class Learner(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        pass


class ReplayBuffer(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        self._buffer = ListReplayBuffer(master=self.manager.replay_buffer)

