import math
from torch import futures
from rlzoo.zoo.role import Role


class Runner(Role):
    def _run(self):
        config = self.kwargs()['config']
        self.sync_model().wait()
        total_learn_steps = config['learner']['total_learn_steps']
        for i in range(total_learn_steps):
            print(f'runstep {i}')
            self.prepare_data().wait()
            print('learn')
            self.learn().wait()            
            self.sync_model(learner=False)

    def prepare_data(self):
        config = self.kwargs()['config']
        capacity = self.remote('replay').rpc_sync().capacity()
        num_worker = config['worker']['num_process']
        num_worker_data = math.ceil(capacity / num_worker)
        return futures.collect_all([
            rref.rpc_async().run_env(num_worker_data)
            for rref in self.remotes('worker')
        ])

    def sync_model(self, worker=True, learner=True):
        self.remote('learner').rpc_sync().push_model()
        rrefs = []
        if worker:
            rrefs += self.remotes('worker')
        if learner:
            rrefs += self.remotes('learner')
        return futures.collect_all([rref.rpc_async().pull_model() for rref in rrefs])
    
    def learn(self):
        return futures.collect_all([
            rref.rpc_async().learn() 
            for rref in self.remotes('learner')
        ])
