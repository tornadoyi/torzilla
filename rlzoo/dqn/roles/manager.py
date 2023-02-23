from torzilla import multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer
from torzilla.rl.parameter_buffer import DictParameterBuffer
from torzilla.rl.tensorboard import SummaryWriter


class Manager(mp.Manager):
    def _start(self):
        config = mp.current_target().kwargs()['config']

        # worker
        cfg = config['worker']
        ns = self.worker = self.Namespace()
        ns.gear = self.Gear(cfg['num_process'])
        ns.queue = self.Queue()

        # evaluator
        cfg = config['eval']
        ns = self.eval = self.Namespace()
        ns.gear = self.Gear(cfg['num_process'])

        # learner
        cfg = config['learner']
        ns = self.learner = self.Namespace()
        ns.gear = self.Gear(cfg['num_process'])

        # replay buffer
        cfg = config['replay']
        self.replay_buffer = ListReplayBuffer(
            capacity=cfg['capacity'], 
        )

        # ps
        cfg = config['ps']
        self.ps = DictParameterBuffer()

        # tb
        cfg = config['tb']['args']
        self.writer = SummaryWriter(**cfg)

    def _exit(self):
        self.worker.gear.close()
        self.eval.gear.close()
        self.learner.gear.close()

        