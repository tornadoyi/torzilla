import time
import random
from torch import futures
from torzilla import multiprocessing as mp
from torzilla import threading, rpc
from rlzoo.zoo import gym


class Worker(mp.Subprocess):
    def _on_start(self, *args, **kwargs):
        self.replay_rrefs = futures.wait_all([
            rpc.rpc_async(info, mp.Process.current_rref) 
            for info in rpc.get_worker_infos() 
            if info.name.startswith('replay_buffer')
        ])

        t_upload = self._start_upload()
        self.manager.worker.gear.join()
        t_upload.join()

    def run_env(self, start_step, end_step):
        return self.manager.worker.gear.apply(
            'run_env',
            start_step,
            end_step
        )

    def pull(self):
        pass

    def _start_upload(self, timeout=0.5):
        Q = self.manager.worker.queue
        datas = []
        def _upload():
            nonlocal datas
            num_data = Q.qsize()
            for _ in range(num_data):
                datas.append(Q.get())

            if len(datas) > 0:
                idx = random.randint(0, len(self.replay_rrefs)-1)
                rb = self.replay_rrefs[idx]
                rb.rpc_sync().put(*datas)
                datas = []
            else:
                time.sleep(timeout)

        def _loop():
            gear = self.manager.worker.gear
            while gear.running:
                _upload()

        t = threading.Thread(target=_loop)
        t.start()
        return t


class Subworker(mp.Subprocess):
    def __init__(self, index, manager, **kwargs) -> None:
        super().__init__(index, manager, **kwargs)
        cfg = self.kwargs['config']['env']
        self._env = gym.make(cfg['id'])
        self._observation = None

    def _on_start(self, *args, **kwargs):
        self.manager.worker.gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        ).join()

    def run_env(self, start_step, end_step):
        Q = self.manager.worker.queue
        agent = self.manager.worker.agent
        
        for step in range(start_step, end_step, 1):
            # reset
            if self._observation is None:
                self._observation, _ = self._env.reset()

            # step
            action = agent.act({
                'observation': self._observation.unsqueeze(0)
            }).squeeze()
            observation, reward, terminated, truncated, info = self._env.step(action)

            # save
            Q.put({
                'step': step,
                'observation': self._observation,
                'next_observation': observation,
                'reward': reward,
                'action': action,
                'done': terminated or truncated,
                'info': info,
            })
            self._observation = None if terminated or truncated else observation
            