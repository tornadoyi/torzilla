import time
import math
from collections import defaultdict
import torch
from torzilla import multiprocessing as mp
from rlzoo.zoo import gym
from rlzoo.zoo.role import Role
from ..agent import Agent

class Worker(Role):
    def _start(self):
        config = self.kwargs()['config']
        ns = self.manager().worker
        env = gym.make(**config['env'])
        cfg = config['agent']
        ns.agent = Agent(env.observation_space, env.action_space, **cfg)
        ns.lock = self.manager().RWLock()

    def run_env(self, total_steps, pull_model=False):
        gear = self.manager().worker.gear
        num_steps = math.ceil(total_steps // gear.connections())

        # push model
        if pull_model:
            self.pull_model()

        # apply
        gear.apply('run_env', args=(num_steps, ))

        # merge
        Q = self.manager().worker.queue
        batches = [Q.get() for _ in range(Q.qsize())]
        datas = {}
        for k in batches[0].keys():
            datas[k] = torch.concat([batch[k] for batch in batches], axis=0)
        
        self.remote('replay').rpc_sync().extend_batch(datas)

    def pull_model(self):
        ns = self.manager().worker
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        with self.manager().worker.lock.wlock():
            ns.agent.load_state_dict(state_dict)
            ns.meta = meta
        
    def close(self):
        self.manager().worker.gear.close()


class Subworker(mp.Target):
    def _run(self):
        config = self.kwargs()['config']
        self.total_learn = config['runner']['num_learn']
        self.env = gym.make(config['env']['id'])
        self.observation = None

        self.manager().worker.gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        )

    def run_env(self, num_steps):
        ns = self.manager().worker
        L, Q = ns.lock, ns.queue
        agent, version = ns.agent, ns.meta['version']
        
        datas = defaultdict(list)
        for _ in range(num_steps):
            # reset
            if self.observation is None:
                self.observation, _ = self.env.reset()

            # step
            with L.rlock():
                eps = agent.calc_eps(self.total_learn)
                action = agent.act({
                    'observation': self.observation.unsqueeze(0)
                }, eps).squeeze()
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # save
            datas['observation'].append(self.observation)
            datas['next_observation'].append(observation)
            datas['reward'].append(reward)
            datas['action'].append(action)
            datas['done'].append(terminated or truncated)
            datas['eps'].append(eps)
            datas['version'].append(torch.tensor(version))
            # datas['info'].append(info)    # low performance
            
            self.observation = None if terminated or truncated else observation
        
        
        batch = dict([(k, torch.stack(v)) for k, v in datas.items()])
        Q.put(batch)