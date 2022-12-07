import random
from torch import futures
from torzilla import multiprocessing as mp
from rlzoo.zoo.role import Role


class Runner(Role):
    def _start(self, *args, **kwargs):

        # workers
        steps, count = 0, 10
        for i in range(1):
            print(f'round {i} start')
            futs = [
                rref.rpc_async().run_env(steps, steps + count)
                for rref in self.remotes('worker')
            ]

            import time
            while True:

                replay_size = self.remote('replay').rpc_sync().size()
                print(f'replay size is {replay_size}')
                
                ok = True
                for f in futs:
                    if f.done(): continue
                    ok = False
                    break
                if ok:
                    break
                time.sleep(1.0)

            time.sleep(2.0)
            replay_size = self.remote('replay').rpc_sync().size()
            print(f'round {i} end with {replay_size}')

            steps += count

        self._terminate()

    
    def _terminate(self):
        for rref in self.remotes('worker'):
            rref.rpc_async().close()