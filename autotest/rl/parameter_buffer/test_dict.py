import unittest
import torch
from torch import futures
import numpy as np
import tempfile
import random
import torzilla.multiprocessing as mp
from torzilla.rl.parameter_buffer import DictParameterBuffer
from torzilla import rpc


class Manager(mp.Manager):
    def _start(self):
        self.buffer = DictParameterBuffer()
        self.Q_send = self.Queue()
        self.Q_recv = self.Queue()

def CreateTarget(name, export_cls):
    dic = {}
    methods = [k for k in export_cls.__dict__.keys() if not k.startswith('_')]
    for meth in methods:
        exec('''def %s(self, /, *args, **kwds):
        return self._core.%s(*args, **kwds)''' % (meth, meth), dic)
    return type(name, (mp.Target,), dic)

class ParameterBuffer(CreateTarget('_ParameterBuffer', DictParameterBuffer)):
    def _start(self):
        self._core = DictParameterBuffer(master=self.manager().buffer)
    
    def size(self):
        return len(self._core)


class TestTarget(mp.Target):
    def _start(self):
        # get all replays
        rank = rpc.get_worker_info().id
        self.ps_rrefs = futures.wait_all([
            rpc.rpc_async(info, mp.current_target_rref) 
            for info in rpc.get_worker_infos() 
            if info.id != rank
        ])
        self.result = None
    
    def _run(self):
        f = getattr(self, self.kwargs()['case'])
        try:
            f()
        except Exception as e:
            if isinstance(e, InterruptedError):
                return
            self.result = e

    def _exit(self):
        self.manager().Q_send.put(self.result)

    def test_ps_put_get(self):
        for i in range(10):
            values, metas = self.init_parameter_buffer()
            datas = futures.wait_all([
                self.ps().rpc_async().get(key=k, value=True, meta=True)
                for k in metas.keys()
            ])
            ps_values, ps_metas = {}, {}
            for i, (value, meta) in enumerate(datas):
                k = list(values.keys())[i]
                ps_values[k], ps_metas[k] = value, meta

            self.check_model(ps_values, values, ps_metas, metas, 'put and get')

    def test_ps_clear(self):
        self.init_parameter_buffer()
        self.ps().rpc_sync().clear()
        self.assertEqual(self.ps().rpc_sync().size(), 0, 'clear')

    def test_ps_keys(self):
        _, metas = self.init_parameter_buffer()
        gt_keys = list(metas.keys())
        keys = self.ps().rpc_sync().keys()
        for key in gt_keys:
            self.assertTrue(key in keys, 'keys')

    def test_ps_remove(self):
        self.init_parameter_buffer()
        gt_keys = list(self.ps().rpc_sync().keys())

        i = random.randint(0, len(gt_keys)-1)
        rk = gt_keys[i]
        del gt_keys[i]
        self.ps().rpc_sync().remove(rk)
        
        keys = self.ps().rpc_sync().keys()
        for key in gt_keys:
            self.assertTrue(key in keys, 'keys')
        
    def test_ps_removes(self):
        self.init_parameter_buffer()
        gt_keys = list(self.ps().rpc_sync().keys())

        ridxes = set(np.random.randint(0, len(gt_keys)-1, size=len(gt_keys) // 2))
        rkeys = [key for i, key in enumerate(gt_keys) if i in ridxes]
        gt_keys = [key for i, key in enumerate(gt_keys) if i not in ridxes]
        self.ps().rpc_sync().removes(rkeys)

        keys = self.ps().rpc_sync().keys()
        for key in gt_keys:
            self.assertTrue(key in keys, 'keys')

    def init_parameter_buffer(self):
        N = 10
        values, metas = {}, {}
        futs = []
        for i in range(random.randint(1, 10)):
            key = str(f'model-{i}')
            # num prameter
            state_dict = {}
            for j in range(random.randint(1, 30)):
                state_dict[str(j)] = torch.rand(size=(N, random.randint(10, 100)))
            meta = {
                'version': random.randint(1, 10),
                'index': i,
            }
            futs.append(self.ps().rpc_async().put(key=key, value=state_dict, meta=meta))
            values[key] = state_dict
            metas[key] = meta

        futures.wait_all(futs)
        return values, metas

    def check_model(self, values=None, gt_values=None, metas=None, gt_metas=None, msg=''):
        if gt_values is not None:
            self.assertEqual(len(values), len(gt_values), msg + ' value length check')
            for k, gt_state_dict in gt_values.items():
                self.assertTrue(k in values, msg + ' value model key check')
                state_dict = values[k]
                for tk, v in gt_state_dict.items():
                    self.assertTrue(tk in state_dict, msg + ' value tensor key check')
                    self.assertTrue(torch.all(state_dict[tk] == v).numpy().tolist(), msg + ' value check')

        if gt_metas is not None:
            self.assertEqual(len(metas), len(gt_metas), msg + ' meta length check')
            for k, gt_meta in gt_metas.items():
                self.assertTrue(k in metas, msg + ' meta model key check')
                meta = metas[k]
                self.assertEqual(len(meta), len(gt_meta), msg + ' meta length check')
                for mk, v in gt_meta.items():
                    self.assertTrue(mk in meta, msg + ' meta key check')
                    self.assertEqual(meta[mk], v, msg + ' meta value check')

    def assertEqual(self, first, second, msg=None):
        Q_send = mp.current_target().manager().Q_send
        Q_recv = mp.current_target().manager().Q_recv
        Q_send.put(('assertEqual', (first, second, msg)))
        if not Q_recv.get():
            raise InterruptedError()

    def assertTrue(self, exp, msg=None):
        Q_send = mp.current_target().manager().Q_send
        Q_recv = mp.current_target().manager().Q_recv
        Q_send.put(('assertTrue', (exp, msg)))
        if not Q_recv.get():
            raise InterruptedError()

    def ps(self):
        idx = random.randint(0, len(self.ps_rrefs)-1)
        return self.ps_rrefs[idx]


class TestDictParameterBuffer(unittest.TestCase):
    
    def setUp(self):
        file = tempfile.NamedTemporaryFile()
        args = [{'target': TestTarget}] + [{'target': ParameterBuffer}] * 1 #random.randint(1, 10)
        self.result = mp.launch_async(
            manager=Manager,
            args = args,
            rpc = {
                'init_method': f'file://{file.name}',
                'world_size': len(args)
            },
            capacity=100,
            case = self._testMethodName
        )

    def tearDown(self) -> None:
        self.result.get()

    def run_test(self):
        Q_send = mp.current_target().manager().Q_send
        Q_recv = mp.current_target().manager().Q_recv
        while True:
            o = Q_send.get()
            if o is None: break
            if isinstance(o, Exception):
                raise o

            fname, args = o
            try:
                getattr(self, fname)(*args)
                Q_recv.put(True)
            except Exception as e:
                Q_recv.put(False)
                raise e

    def test_ps_put_get(self):
        self.run_test()

    def test_ps_clear(self):
        self.run_test()

    def test_ps_keys(self):
        self.run_test()

    def test_ps_remove(self):
        self.run_test()

    def test_ps_removes(self):
        self.run_test()



if __name__ == '__main__':
    unittest.main()