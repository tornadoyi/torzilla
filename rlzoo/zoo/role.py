import random
from torch import futures
from torzilla import multiprocessing as mp
from torzilla import rpc

class Role(mp.Target):
    def remotes(self, name):
        param_name = f'__remotes__{name}__'
        rrefs = getattr(self, param_name, None)
        if rrefs is None:
            rrefs = futures.wait_all([
                rpc.rpc_async(info, mp.current_target_rref) 
                for info in rpc.get_worker_infos() 
                if info.name.startswith(name)
            ])
            if len(rrefs) == 0:
                raise AttributeError(f'rpc worker with prefix "{name}" is not exist')
            setattr(self, param_name, rrefs)
        return rrefs

    def remote(self, name, idx=None):
        rrefs = self.remotes(name)
        idx = idx or random.randint(0, len(rrefs)-1)
        return rrefs[idx]