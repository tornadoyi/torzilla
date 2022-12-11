from torzilla import multiprocessing as mp
from torzilla.rl.replay_buffer import ListReplayBuffer
from torzilla.rl.parameter_buffer import DictParameterBuffer
from rlzoo.zoo import gym
from ..agent import Agent

class Manager(mp.Manager):
    def _start(self):
        config = mp.current_target().kwargs()['config']

        # worker
        cfg = config['worker']
        ns = self.worker = self.Namespace()
        ns.gear = self.Gear(cfg['num_process'])
        ns.queue = self.Queue()

        # worker's agent
        cfg = config['agent']
        env = gym.make(**config['env'])
        ns.agent = Agent(env.observation_space, env.action_space, **cfg)

        # learner
        cfg = config['learner']
        ns = self.learner = self.Namespace()
        ns.gear = self.Gear(cfg['num_process'])

        # replay buffer
        cfg = config['replay_buffer']
        self.replay_buffer = ListReplayBuffer(
            capacity=cfg['capacity'], 
        )

        # ps
        cfg = config['ps']
        self.ps = DictParameterBuffer()

        