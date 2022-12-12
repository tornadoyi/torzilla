import tempfile
import itertools
import torzilla as tz
from torzilla import multiprocessing as mp
from rlzoo.dqn.processes import *

A = tz.Argument

_BATCH_SIZE = 32

CONFIG = dict(
    env = dict(
        id = A(type=str, default='CartPole-v1')
    ),

    worker = dict(
        num_process = A(type=int, default=5),
    ),

    learner = dict(
        num_process = A(type=int, default=2),
        distributed = dict(
            backend = A(type=str, default='gloo')
        ),
        optimizer = dict(
            optim = dict(
                name = A(type=str, default='RMSprop'),
                lr = A(type=float, default=1e-5),
            ),
            max_grad_norm = A(type=float, default=None),
        ),
        batch_size = A(type=int, default=_BATCH_SIZE,),
        total_learn_steps = A(type=int, default=3,),
    ),
    
    replay = dict(
        num_process = A(type=int, default=2),
        capacity = A(type=int, default=_BATCH_SIZE),
    ),

    ps = dict(
        num_process = A(type=int, default=1),
    ),

    agent = dict(
        double_q = A(action='store_true', default=False),
        gamma = A(type=float, default=0.99),
        eps = A(type=float, default=0.1),
        eps_annealing = A(type=float, default=0),
        qtarget_update_freq = A(type=int, default=100),
        q_func_args = dict(
            hiddens = A(type=int, nargs='+', default=[256]),
            dueling = A(action='store_true', default=False),
        ),
    )
)

def prepare_roles(config):
    file = tempfile.NamedTemporaryFile()
    rpc_args = {
        'init_method': f'file://{file.name}'
    }
    procs = []

    it_rank = itertools.count()
    def get_rank(): return next(it_rank)
    
    # runner
    procs.append(dict(
        target = Runner,
        rpc = dict(
            rank = get_rank(),
            name = 'runner-0',
            **rpc_args
        ),
    ))

    # worker
    procs.append(dict(
        target = Worker,
        rpc = dict(
            rank = get_rank(),
            name = f'worker-0',
            **rpc_args
        ),
    ))
    for _ in range(config['worker']['num_process']):
        procs.append(dict(
            target = Subworker,
        ))

    # learner
    leaner_file = tempfile.NamedTemporaryFile()
    cfg = config['learner']
    for i in range(cfg['num_process']):
        procs.append(dict(
            target = Learner,
            rpc = dict(
                rank = get_rank(),
                name = f'learner-{i}',
                **rpc_args,
            ),
            distributed = dict(
                rank = i,
                init_method = f'file://{leaner_file.name}',
                world_size = cfg['num_process'],
                **cfg['distributed']
            )
        ))

    # replay buffer
    for i in range(config['replay']['num_process']):
        procs.append(dict(
            target = ReplayBuffer,
            rpc = dict(
                rank = get_rank(),
                name = f'replay-{i}',
                **rpc_args
            ),
        ))

    # parameter server
    for i in range(config['ps']['num_process']):
        procs.append(dict(
            target = ParameterServer,
            rpc = dict(
                rank = get_rank(),
                name = f'ps-{i}',
                **rpc_args
            ),
        ))

    world_size = get_rank()
    for arg in procs:
        rpc = arg.get('rpc', None)
        if rpc is None: continue
        rpc['world_size'] = world_size

    return procs


def main():
    config = tz.parse_hargs(CONFIG)
    args = prepare_roles(config)
    mp.lanuch(
        args = args,
        manager = Manager,
        config = config
    )
    

if __name__ == '__main__':
    main()