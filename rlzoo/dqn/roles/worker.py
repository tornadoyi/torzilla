import time
import math
import torch
from torzilla import multiprocessing as mp
from torzilla import threading
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
        Q = self.manager().worker.queue
        gear = self.manager().worker.gear
        num_steps = math.ceil(total_steps // gear.connections())

        # upload thread
        def _batch_upload():
            remain_data = num_steps * gear.connections()
            while remain_data > 0:
                qsize = Q.qsize()
                if qsize == 0:
                    time.sleep(0.5)
                    continue
                datas = [Q.get() for _ in range(qsize)]
                self.remote('replay').rpc_sync().extend(datas)
                remain_data -= len(datas)

        # push model
        if pull_model:
            self.pull_model()

        # start thread
        t = threading.Thread(target=_batch_upload)
        t.start()
        
        # apply
        gear.apply('run_env', args=(num_steps, ))
        t.join()

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
        Q = self.manager().worker.queue
        L = self.manager().worker.lock
        ns = self.manager().worker
        agent, version = ns.agent, ns.meta['version']
            
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
            Q.put({
                'observation': self.observation,
                'next_observation': observation,
                'reward': reward,
                'action': action,
                'done': terminated or truncated,
                'info': info,
                'eps': eps,
                'sample_version': torch.tensor(version),
            })
            self.observation = None if terminated or truncated else observation