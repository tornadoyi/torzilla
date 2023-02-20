import time
import torch
import torch.distributed as dist
from torzilla import threading, rpc
from torzilla.distributed.optim import ReducedOptimizer
from rlzoo.zoo import gym
from rlzoo.zoo.role import Role
from rlzoo.dqn.agent import Agent
from rlzoo.dqn.roles.tensorboard import Tensorboard


class Learner(Role):
    def _start(self):
        kwargs = self.kwargs()
        config = kwargs['config']

        # init distributed
        dist.init_process_group(**kwargs['distributed'])

        # agent
        env = gym.make(**config['env'])
        self.agent = Agent(env.observation_space, env.action_space, **config['agent'])
        if torch.cuda.is_available():
            self.agent = self.agent.cuda(dist.get_rank() % torch.cuda.device_count())

        # optimizer
        cfg = config['learner']['optimizer']
        optim_cls = getattr(torch.optim, cfg['optim']['name'])
        optimizer = optim_cls(
            [p for p in self.agent.parameters() if p.requires_grad], 
            **cfg['optim']['args']
        )
        self.optimizer = ReducedOptimizer(optimizer)

        # scheduler
        sched_cls = getattr(torch.optim.lr_scheduler, cfg['scheduler']['name'])
        self.scheduler = sched_cls(optimizer, **cfg['scheduler']['args'])

        # meta
        self.meta = {'version': 0}

        # lock
        self.lock = threading.RLock()

    def push_model(self):
        with self.lock:
            self.remote('ps').rpc_sync().put(
                value = dict([(k, v.cpu()) for k, v in self.agent.state_dict().items()]),
                meta = self.meta
            )

    def pull_model(self):
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        with self.lock:
            self.agent.load_state_dict(state_dict)
            self.meta = meta

    def learn(self, master_name, print_tb, push_model):
        config = self.kwargs()['config']
        cfg = config['learner']
        batch_size = cfg['batch_size']
        is_master = rpc.get_worker_info().name == master_name
        print_tb, push_model = is_master and print_tb, is_master and push_model

        # sample
        datas = self.remote('replay').rpc_sync().sample(batch_size)
        
        inputs = {}
        for k, v in datas[0].items():
            if not isinstance(v, torch.Tensor): continue
            inputs[k] = torch.vstack([d[k] for d in datas]).squeeze()

        if torch.cuda.is_available():
            index = dist.get_rank() % torch.cuda.device_count()
            inputs = dict([(k, v.cuda(index)) for k, v in inputs.items()])


        # grad norm
        def _grad_norm():
            max_grad_norm = cfg['optimizer']['max_grad_norm']
            if max_grad_norm is None: return
            params = self.optimizer.optimizer.param_groups[0]['params']
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        # loss & optimize
        with self.lock:
            # loss
            self.optimizer.zero_grad()
            learn_info = self.agent.learn(inputs)
            loss = learn_info['loss']

            # optimize
            loss.backward()
            self.optimizer.step(_grad_norm)
            self.scheduler.step()

            # meta
            self.meta.update(dict(
                version = self.agent.num_learn.cpu().numpy().tolist(),
                timestamp = time.time()
            ))

            # print tb
            if print_tb: 
                self._print_tb(learn_info, inputs)

            if push_model:
                self.push_model()


    def _print_tb(self, learn_info, inputs):
        # to cpu
        learn_info = dict([(k, v.cpu()) for k, v in learn_info.items()])
        inputs = dict([(k, v.cpu()) for k, v in inputs.items()])

        fut_name = '__fut_print_tb__'
        f_print = getattr(self, fut_name, None)
        if f_print is not None:
            f_print.wait()

        # version
        version = self.meta['version']
        
        # learn result
        idcts = []
        for k, v in learn_info.items():
            idcts += Tensorboard.guess(f'learner/{k}', v, version)

        # optimizer
        idcts += Tensorboard.guess(f'learner/lr', self.scheduler.get_last_lr(), version)
        
        # grad
        params = self.optimizer.optimizer.param_groups[0]['params']
        device = params[0].grad.device
        norm_grads = torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in params]).cpu()
        total_norm_grad = torch.norm(norm_grads, 2.0)
        idcts.append(('add_histogram', ('learner/grad_norm_dist', norm_grads, version)))
        idcts.append(('add_scalar', ('learner/grad_norm', total_norm_grad, version)))

        # inputs
        for k, v in inputs.items():
            idcts += Tensorboard.guess(f'input/{k}', v, version)
        
        idcts += Tensorboard.guess(f'input/off_version', version - inputs['version'], version)

        f_print = self.remote('tb').rpc_async().adds(idcts)
        setattr(self, fut_name, f_print)
        