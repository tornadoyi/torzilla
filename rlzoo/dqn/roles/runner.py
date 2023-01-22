import math
import time
from functools import partial
from torch import futures
from rlzoo.zoo.role import Role


class Runner(Role):
    def _run(self):
        # config
        config = self.kwargs()['config']
        cfg_run = config['runner']
        num_learn = cfg_run['num_learn']
        freq_push_model = cfg_run['freq_push_model']
        freq_learn_print = cfg_run['freq_learn_print']
        freq_eval = cfg_run['freq_eval']
        freq_sample = cfg_run['freq_sample']
        num_sampled_data = cfg_run['num_sampled_data']
        
        # sync model
        self.remote('learner').rpc_sync().push_model()
        (self.remotes('worker') + self.remotes('learner')).rpc_sync().pull_model()

        # init replay buffer
        replay_cap = self.remote('replay').rpc_sync().capacity()
        num_sample_per_worker = math.ceil(replay_cap / len(self.remotes('worker')))
        self.remotes('worker').rpc_sync().run_env(num_sample_per_worker)

        # main loop
        for version in range(1, num_learn, 1):
            print(f'current version: {version}')
            
            # prepare data
            if version % freq_sample == 0:
                self._run_env(version, num_sampled_data)

            # eval
            if version % freq_eval == 0:
                self._eval(version)

            # learn
            push_model = version % freq_push_model == 0
            print_tb = version % freq_learn_print == 0
            self._learn(version, push_model, print_tb).wait()

    def _learn(self, version, push_model, print_tb):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/learn_cost', cost, global_step=version)

        # learn
        master = 'learner-0'
        fut = self.remotes('learner').rpc_async().learn(master, print_tb, push_model)
        fut.add_done_callback(partial(_finish, time.time(), version))
        return fut

    def _run_env(self, version, num_data):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/run_env_cost', cost, global_step=version)

        param_name = '__f_run_env__'
        fut = getattr(self, param_name, None)
        if fut is not None and not fut.done():
            return

        num_data_per_worker = math.ceil(num_data / len(self.remotes('worker')))
        fut = self.remotes('worker').rpc_async().run_env(num_data_per_worker, pull_model=True)
        fut.add_done_callback(partial(_finish, time.time(), version))
        setattr(self, param_name, fut)
        return fut

    def _eval(self, version):
        def _finish(start_time, version, *args):
            cost = time.time() - start_time
            self.remote('tb').rpc_async().add_scalar('runner/eval_cost', cost, global_step=version)
        
        param_name = '__f_eval__'
        fut = getattr(self, param_name, None)
        if fut is not None and not fut.done():
            return

        fut = self.remote('eval').rpc_async().evaluate()
        fut.add_done_callback(partial(_finish, time.time(), version))
        setattr(self, param_name, fut)
        return fut
        