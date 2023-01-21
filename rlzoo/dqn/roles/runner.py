import math
import time
from functools import partial
from torch import futures
from rlzoo.zoo.role import Role


class Runner(Role):
    def _run(self):
        # config
        config = self.kwargs()['config']
        self.num_learn = config['runner']['num_learn']
        self.num_learn_push_model = config['runner']['num_learn_push_model']
        self.num_learn_eval = config['runner']['num_learn_eval']
        self.num_learn_sample = config['runner']['num_learn_sample']
        self.num_sample = config['runner']['num_sample']
        self.replay_cap = self.remote('replay').rpc_sync().capacity()

        # sync model
        self.remote('learner').rpc_sync().push_model()
        futures.wait_all([
            rref.rpc_async().pull_model()
            for rref in self.remotes('worker') + self.remotes('learner')
        ])

        # init replay buffer
        num_sample_per_worker = math.ceil(self.replay_cap / len(self.remotes('worker')))
        futures.wait_all([
            rref.rpc_async().run_env(num_sample_per_worker)
            for rref in self.remotes('worker')
        ])

        # start to learn
        futures.wait_all([
            rref.rpc_async().start_learn()
            for rref in self.remotes('learner')
        ])
        
        # learn loop
        for version in range(1, self.num_learn, 1):
            # if version % 100 == 0:
            print(f'current version: {version}')
            
            # prepare data
            if version % self.num_learn_sample == 0:
                self._run_env(version)

            # eval
            if version % self.num_learn_eval == 0:
                self._eval(version)

            # learn
            self._learn(version).wait()

        # stop to learn
        futures.wait_all([
            rref.rpc_async().stop_learn()
            for rref in self.remotes('learner')
        ])

    def _learn(self, version):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/learn_cost', cost, global_step=version)
        
        # push model tag
        push_model = (version % self.num_learn_push_model == 0)

        # learn
        f_learn = futures.collect_all([
            rref.rpc_async().next_learn(version, push_model=(i==0 and push_model), print_tb=(i==0))
            for i, rref in enumerate(self.remotes('learner'))
        ])
        f_learn.add_done_callback(partial(_finish, time.time(), version))
        return f_learn

    def _run_env(self, version):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/run_env_cost', cost, global_step=version)

        num_sample_per_worker = math.ceil(self.num_sample / len(self.remotes('worker')))

        f_run_env = getattr(self, '_f_run_env_', None)
        if f_run_env is not None and not f_run_env.done():
            return
        f_run_env = self._f_run_env_ = futures.collect_all([
            rref.rpc_async().run_env(num_sample_per_worker, pull_model=True)
            for rref in self.remotes('worker')
        ])
        f_run_env.add_done_callback(partial(_finish, time.time(), version))
        return f_run_env

    def _eval(self, version):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/eval_cost', cost, global_step=version)
            
        f_eval = getattr(self, '_f_eval_', None)
        if f_eval is not None and not f_eval.done():
            return

        f_eval = self._f_eval_ = futures.collect_all([
            rref.rpc_async().evaluate()
            for rref in self.remotes('eval')
        ])
        f_eval.add_done_callback(partial(_finish, time.time(), version))
        return f_eval
        