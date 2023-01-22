import tempfile
import itertools
import torzilla as tz
from torzilla import multiprocessing as mp
from rlzoo.dqn.roles import *

A = tz.Argument

_BATCH_SIZE = 128
_TOTAL_LEARN = int(1e4)

CONFIG = dict(

    runner = dict(
        num_learn = A(type=int, default=_TOTAL_LEARN),
        freq_push_model = A(type=int, default=1),
        freq_learn_print = A(type=int, default=3),
        freq_eval = A(type=int, default=10),
        freq_sample = A(type=int, default=1),
        num_sampled_data = A(type=int, default=_BATCH_SIZE // 3),
    ),

    env = dict(
        id = A(type=str, default='CartPole-v1'),
        enable_id_wrapper = A(action='store_true', default=True),
    ),

    worker = dict(
        num_process = A(type=int, default=5),
    ),

    eval = dict(
        num_process = A(type=int, default=1),
    ),

    learner = dict(
        num_process = A(type=int, default=2),
        distributed = dict(
            backend = A(type=str, default='gloo')
        ),
        optimizer = dict(
            optim = dict(
                name = A(type=str, default='RMSprop'),
                args = dict(
                    lr = A(type=float, default=1e-3),
                )
            ),
            scheduler = dict(
                name = A(type=str, default='LinearLR'),
                args = dict(
                    start_factor = A(type=float, default=1.0),
                    end_factor = A(type=float, default=0.1),
                    total_iters = A(type=int, default=_TOTAL_LEARN // 3),
                )
            ),
            max_grad_norm = A(type=float, default=None),
        ),
        batch_size = A(type=int, default=_BATCH_SIZE,),
    ),
    
    replay = dict(
        num_process = A(type=int, default=1),
        capacity = A(type=int, default=_BATCH_SIZE * 10),
    ),

    ps = dict(
        num_process = A(type=int, default=1),
    ),

    tb = dict(
        num_process = A(type=int, default=1),
        args = dict(
            log_dir = A(type=str, default='runs/dqn'),
            # flush_secs = A(type=float, default=3.0),
        )
    ),

    agent = dict(
        double_q = A(action='store_true', default=False),
        gamma = A(type=float, default=0.99),
        eps = A(type=float, default=0.3),
        eps_annealing = A(type=float, default=5.0),
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
        'init_method': f'file://{file.name}',
        # 'rpc_timeout': 30,

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

    # evaluator
    procs.append(dict(
        target = Evaluator,
        rpc = dict(
            rank = get_rank(),
            name = f'eval-0',
            **rpc_args
        ),
    ))
    for _ in range(config['eval']['num_process']):
        procs.append(dict(
            target = SubEvaluator,
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
                **rpc_args
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

    # tb
    for i in range(config['tb']['num_process']):
        procs.append(dict(
            target = Tensorboard,
            rpc = dict(
                rank = get_rank(),
                name = f'tb-{i}',
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