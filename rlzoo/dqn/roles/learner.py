import copy
import time
import torch
import torch.distributed as dist
from torzilla import threading
from torzilla.distributed.optim import ReducedOptimizer
from rlzoo.zoo import gym
from rlzoo.zoo.role import Role
from rlzoo.dqn.agent import Agent


class Learner(Role):
    def _start(self):
        kwargs = self.kwargs()
        config = kwargs['config']

        # init distributed
        dist.init_process_group(**kwargs['distributed'])

        # agent
        env = gym.make(**config['env'])
        self.agent = Agent(env.observation_space, env.action_space, **config['agent'])

        # optimizer
        cfg = config['learner']['optimizer']
        optim_args = copy.copy(cfg['optim'])
        name = optim_args['name']
        del optim_args['name']
        optim_cls = getattr(torch.optim, name)
        self.optimizer = ReducedOptimizer(optim_cls(
            [p for p in self.agent.parameters() if p.requires_grad], 
            **optim_args
        ))

        # learn info
        self.learn_info = {
            'thread': None,
            'target_version': 0, 
            'running': False,
            'meta': {
                'version': 0
            }
        }

        # cond
        self.cond = threading.Condition()

    def push_model(self):
        with self.cond:
            state_dict, meta = self.agent.state_dict(), self.learn_info['meta']
        return self._push_model(state_dict, meta)

    def pull_model(self):
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        with self.cond:
            self.agent.load_state_dict(state_dict)
            self.learn_info['meta'] = meta

    def start_learn(self, target_version):
        info = self.learn_info
        
        def valid_target_version():
            return not info['running'] or info['meta']['version'] < info['target_version']

        def _loop():
            while info['running']:
                with self.cond:
                    self.cond.wait_for(valid_target_version)
                    if not info['running']:
                        info['thread'] = None
                        break
                self._learn_once()

        with self.cond:
            if info['thread']:
                raise RuntimeError('repeated learn task')
            info['thread'] = threading.Thread(target=_loop)
            info['target_version'] = target_version
            info['running'] = True
            info['thread'].start()
            self.cond.notify()

    def stop_learn(self):
        with self.cond:
            t = self.learn_info['thread']
            if t is None: return
            self.learn_info['running'] = False
            self.cond.notify()
        t.join()

    def next_learn(self, target_version, wait_step, push_model=False):
        info = self.learn_info
        def _cond():
            return info['meta']['version'] >= wait_step or not info['running']

        with self.cond:
            info['target_version'] = target_version
            self.cond.notify_all()
            self.cond.wait_for(_cond)
            if push_model:
                state_dict, meta = self.agent.state_dict(), self.learn_info['meta']
        if push_model:
            self._push_model(state_dict, meta)

    def _learn_once(self):
        config = self.kwargs()['config']
        cfg = config['learner']
        batch_size = cfg['batch_size']

        # sample
        datas = self.remote('replay').rpc_sync().sample(batch_size)
        inputs = {}
        for k, v in datas[0].items():
            if not isinstance(v, torch.Tensor): continue
            inputs[k] = torch.vstack([d[k] for d in datas]).squeeze()

        # grad norm
        def _grad_norm():
            max_grad_norm = cfg['optimizer']['max_grad_norm']
            if max_grad_norm is None: return
            params = self.optimizer.optimizer.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        # loss & optimize
        with self.cond:
            self.optimizer.zero_grad()
            loss = self.agent.learn(inputs)
            loss.backward()
            self.optimizer.step(_grad_norm)
            self.learn_info['meta'].update(dict(
                version = self.agent.num_learn.numpy().tolist(),
                timestamp = time.time()
            ))
            self.cond.notify_all()

    def _push_model(self, state_dict, meta, asyn=False):
        r = self.remote('ps').rpc_async().put(
            value = state_dict,
            meta = meta
        )
        return r if asyn else r.wait()