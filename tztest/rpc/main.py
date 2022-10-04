import argparse
import torzilla
from torzilla.launcher import process
from torzilla import rpc

N = 5

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
    parser.add_argument('--world-size', type=int, default=N)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num-process', type=int, default=N)
    parser.add_argument('--num-rpc', type=int, default=None)
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:3999')
    parser.add_argument('--num-worker-threads', type=int, default=None)
    return parser.parse_args()

class MainProcess(process.MainProcess):
    pass


class Subprocess(process.Subprocess):
    def _on_start(self):
        self._try_init_rpc()
        
        if rpc.is_init():
            info = rpc.get_worker_info()
            print(f'proc {self.proc_index} init rpc finish, name={info.name} id={info.id}')
        else:
            print(f'proc {self.proc_index} start')

        if rpc.is_init(): rpc.shutdown()
        

def main():
    args = parse_args()

    torzilla.lanuch(
        num_process=args.num_process,
        mainproc='tztest.rpc.main:MainProcess',
        subproc='tztest.rpc.main:Subprocess',
        rpc = dict(
            world_size=args.world_size,
            rank=args.rank,
            num_rpc=args.num_rpc or args.world_size,
            init_method=args.init_method,
            num_worker_threads=args.num_worker_threads,
        ),
    )

if __name__ == '__main__':
    main()