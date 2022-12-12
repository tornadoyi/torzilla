import time
from torch import futures
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

    def run_env(self, num_steps):
        Q = self.manager().worker.queue
        gear = self.manager().worker.gear

        # upload thread
        def _batch_upload():
            remain_data = num_steps * gear.connections()
            while remain_data > 0:
                # print(f'remain_data: {remain_data}')
                qsize = Q.qsize()
                if qsize == 0:
                    time.sleep(0.5)
                    continue
                datas = [Q.get() for _ in range(qsize)]
                self.remote('replay').rpc_sync().extend(datas)
                remain_data -= len(datas)

        # start thread
        t = threading.Thread(target=_batch_upload)
        t.start()
        
        # apply
        gear.apply('run_env', num_steps)
        t.join()

    def pull_model(self):
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        agent = self.manager().worker.agent
        with self.manager().worker.lock.wlock():
            agent.load_state_dict(state_dict)

    def close(self):
        self.manager().worker.gear.close()


class Subworker(mp.Target):
    def _run(self):
        cfg = self.kwargs()['config']['env']
        self._env = gym.make(cfg['id'])
        self._observation = None

        self.manager().worker.gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        )

    def run_env(self, num_steps):
        Q = self.manager().worker.queue
        L = self.manager().worker.lock
        agent = self.manager().worker.agent
            
        for _ in range(num_steps):
            # reset
            if self._observation is None:
                self._observation, _ = self._env.reset()

            # step
            with L.rlock():
                action = agent.act({
                    'observation': self._observation.unsqueeze(0)
                }).squeeze()
            observation, reward, terminated, truncated, info = self._env.step(action)
            
            # save
            Q.put({
                'observation': self._observation,
                'next_observation': observation,
                'reward': reward,
                'action': action,
                'done': terminated or truncated,
                'info': info,
            })
            self._observation = None if terminated or truncated else observation