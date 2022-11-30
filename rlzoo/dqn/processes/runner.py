import random
from torch import futures
from torzilla import multiprocessing as mp
from torzilla import threading, rpc


class Runner(mp.Subprocess):
    def _on_start(self, *args, **kwargs):

        # workers
        steps, count = 0, 10
        for i in range(10):
            print(f'round {i} start')
            futs = [
                rref.rpc_async().run_env(steps, steps + count)
                for rref in self._workers()
            ]

            import time
            while True:

                replay_size = self._replay().rpc_sync().size()
                print(f'replay size is {replay_size}')
                
                ok = True
                for f in futs:
                    if f.done(): continue
                    ok = False
                    break
                if ok:
                    break
                time.sleep(1.0)

            steps += count
  
    def _roles(self, name):
        param_name = f'_{name}'
        roles = getattr(self, param_name, None)
        if roles is None:
            roles = futures.wait_all([
                rpc.rpc_async(info, mp.Process.current_rref) 
                for info in rpc.get_worker_infos() 
                if info.name.startswith(name)
            ])
            setattr(self, param_name, roles)
        return roles

    def _replays(self):
        return self._roles('replay_buffer')

    def _replay(self):
        replays = self._replays()
        idx = random.randint(0, len(replays)-1)
        return replays[idx]

    def _workers(self):
        return self._roles('worker')
        
    def _learner(self):
        return self._roles('learner')
            
