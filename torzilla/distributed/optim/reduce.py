from torch import distributed as dist

class ReducedOptimizer(object):
    def __init__(self, optimizer, method='mean'):
        if method not in ('mean', 'sum'):
            raise ValueError(f'unsupported reduce method {method}')

        self._optimizer = optimizer
        self._method = method

    @property
    def optimizer(self):
        return self._optimizer

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def step(self, closure):
        if self._method == 'mean':
            self._reduce_mean()
        else:
            self._reduce_sum()
        self._optimizer.step(closure)

    def _reduce_sum(self):
        handlers = []
        for group in self._optimizer.param_groups:
            for param in group['params']:
                h = dist.all_reduce(param.grad.data, dist.ReduceOp.SUM, async_op=True)
                handlers.append(h)
        for h in handlers: 
            h.wait()

    def _reduce_mean(self):
        size = float(dist.get_world_size())
        self._reduce_sum()
        for group in self._optimizer.param_groups:
            for param in group['params']:
                param.grad.data /= size
