import tempfile
import itertools
from collections import OrderedDict as odcit
import torzilla as tz
from torzilla import multiprocessing as mp
from rlzoo.dqn.processes import *

A = tz.Argument

_BATCH_SIZE = 1024

CONFIG = odcit(
    env = odcit(
        id = A(type=str, default='CartPole-v1')
    ),

    worker = odcit(
        num_process = A(type=int, default=10),
    ),

    learner = odcit(
        num_process = A(type=int, default=3),
    ),
    
    replay_buffer = odcit(
        num_process = A(type=int, default=3),
        capacity = A(type=int, default=_BATCH_SIZE*3),
        max_cache_size = A(type=int, default=_BATCH_SIZE*3),
    ),

    agent = odcit(
        double_q = A(action='store_true', default=False),
        gamma = A(type=float, default=0.99),
        q_func_args = odcit(
            hiddens = A(type=int, nargs='+', default=[256]),
            dueling = A(action='store_true', default=False),
        )
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
        subproc = Runner,
        rpc = dict(
            rank = get_rank(),
            name = 'runner-0',
            **rpc_args
        ),
        config = config
    ))

    # worker
    procs.append(dict(
        subproc = Worker,
        rpc = dict(
            rank = get_rank(),
            name = f'worker-0',
            **rpc_args
        ),
        config = config
    ))
    for _ in range(config['worker']['num_process']):
        procs.append(dict(
            subproc = Subworker,
            config = config
        ))

    # replay buffer
    for i in range(config['replay_buffer']['num_process']):
        procs.append(dict(
            subproc = ReplayBuffer,
            rpc = dict(
                rank = get_rank(),
                name = f'replay_buffer-{i}',
                **rpc_args
            ),
            config = config
        ))

    world_size = get_rank()
    for arg in procs:
        rpc = arg.get('rpc', None)
        if rpc is None: continue
        rpc['world_size'] = world_size

    return procs


def main():
    args = tz.parse_hargs(CONFIG)
    subproc_args = prepare_roles(args)
    mp.lanuch(
        subproc_args = subproc_args,
        manager = Manager,
        config = args
    )
    

if __name__ == '__main__':
    main()