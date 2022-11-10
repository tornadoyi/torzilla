from torch.distributed import rpc as _rpc
from torzilla.core import utility as U

def init_rpc(
    rank,
    world_size,
    backend=None,
    name=None,
    init_method=None, 
    rpc_timeout=None,
    num_worker_threads=None,
    device_maps=None,
    devices=None,

):
    U.assert_type(world_size, int)
    U.assert_type(rank, int)
    U.assert_type(backend, str, null=True)
    U.assert_type(init_method, str, null=True)
    U.assert_type(rpc_timeout, int, float, null=True)
    U.assert_type(num_worker_threads, int, null=True)
    U.assert_type(name, str, null=True)
    name = f'{rank}' if name is None else name

    # option
    if backend is None or backend == _rpc.BackendType.TENSORPIPE:
        args = {
            'num_worker_threads': num_worker_threads,
            'rpc_timeout': rpc_timeout,
            'init_method': init_method,
            'device_maps': device_maps,
            'devices': devices,
        }
        opt_args = U.pick_args(args, args.keys(), drop_none=True)
        options = _rpc.TensorPipeRpcBackendOptions(**opt_args)
    else:
        args = {
            'rpc_timeout': rpc_timeout,
            'init_method': init_method,
        }
        opt_args = U.pick_args(args, args.keys(), drop_none=True)
        options = _rpc.RpcBackendOptions(**opt_args)

    return _rpc.init_rpc(
        name=name,
        backend=backend,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )


def is_init():
    try:
        _rpc.get_worker_info()
        return True
    except:
        return False