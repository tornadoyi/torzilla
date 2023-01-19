import math
from torch import futures
from rlzoo.zoo.role import Role


class Runner(Role):
    def _run(self):
        config = self.kwargs()['config']
        num_learn = config['learner']['num_learn']
        off_version = config['learner']['off_version']
        replay_cap = self.remote('replay').rpc_sync().capacity()

        # sync model
        self.remote('learner').rpc_sync().push_model()
        futures.wait_all([
            rref.rpc_async().pull_model()
            for rref in self.remotes('worker') + self.remotes('learner')
        ])

        # init replay buffer
        futures.wait_all([
            rref.rpc_async().run_env(replay_cap)
            for rref in self.remotes('worker')
        ])

        # start to learn
        gen_step_per_learn = replay_cap / off_version
        futures.wait_all([
            rref.rpc_async().start_learn(1)
            for rref in self.remotes('learner')
        ])
        
        # learn loop
        for version in range(1, num_learn+1, 1):
            print(version)
            # wait current version learn finish
            futures.wait_all([
                rref.rpc_async().next_learn(version+1, version, push_model=i==0)
                for i, rref in enumerate(self.remotes('learner'))
            ])

            # prepare data
            futures.wait_all([
                rref.rpc_async().run_env(gen_step_per_learn, pull_model=True)
                for rref in self.remotes('worker')
            ])

        # stop to learn
        futures.wait_all([
            rref.rpc_async().stop_learn()
            for rref in self.remotes('learner')
        ])
