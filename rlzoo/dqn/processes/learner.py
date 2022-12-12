import copy
import torch
import torch.distributed as dist
from torzilla import threading
from torzilla.distributed.optim import ReducedOptimizer
from rlzoo.zoo.role import Role
from rlzoo.zoo import gym
from ..agent import Agent


class Learner(Role):

    def _start(self):
        kwargs = self.kwargs()
        config = kwargs['config']

        # init distributed
        dist.init_process_group(**kwargs['distributed'])

        # agent
        env = gym.make(**config['env'])
        cfg = config['agent']
        self.agent = Agent(env.observation_space, env.action_space, **cfg)

        # optimizer
        cfg = config['learner']['optimizer']
        optim_args = copy.copy(cfg['optim'])
        name = optim_args['name']
        del optim_args['name']
        optim_cls = getattr(torch.optim, name)
        self.optimizer = ReducedOptimizer(optim_cls(
            [p for p in self.agent.parameters() if p.requires_grad], 
            **optim_args
        ))

        # lock
        self.lock = threading.RWLock()

    def close(self):
        self.manager().learner.gear.close()

    def push_model(self, meta=None):
        with self.lock.rlock():
            state_dict = self.agent.state_dict()

        self.remote('ps').rpc_sync().put(
            value = state_dict,
            meta = meta
        )

    def pull_model(self):
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        with self.lock.wlock():
            self.agent.load_state_dict(state_dict)

    def learn(self, batch_size=None):
        config = self.kwargs()['config']
        cfg = config['learner']
        batch_size = batch_size or cfg['batch_size']

        print('sampel')
        # sample
        datas = self.remote('replay').rpc_sync().sample(batch_size)
        inputs = {}
        for k, v in datas[0].items():
            if not isinstance(v, torch.Tensor): continue
            inputs[k] = torch.vstack([d[k] for d in datas]).squeeze()

        # grad norm
        def _grad_norm():
            max_grad_norm = cfg['optimizer']['max_grad_norm']
            if max_grad_norm is None: return
            params = self.optimizer.optimizer.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        # loss & optimize
        self.optimizer.zero_grad()
        print('learn start')
        loss = self.agent.learn(inputs)
        print('learn end')
        loss.backward()

        for k, p in self.agent.named_parameters():
            if not p.requires_grad or p.grad is not None: continue
            print(f'{k} no grad')

        self.optimizer.step(_grad_norm)
