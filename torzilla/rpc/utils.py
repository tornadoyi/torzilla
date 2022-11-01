from torch.distributed import rpc as _rpc

def rpc_agent():
    return _rpc.api._get_current_rpc_agent()

def barrier():
    return _rpc.api._all_gather(None, timeout=0)

def get_worker_infos():
    return rpc_agent().get_worker_infos()