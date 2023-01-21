from rlzoo.zoo import gym
from rlzoo.zoo.role import Role
from torzilla import multiprocessing as mp
from .tensorboard import Tensorboard
from ..agent import Agent


class Evaluator(Role):
    def _start(self):
        config = self.kwargs()['config']
        ns = self.manager().eval
        env = gym.make(**config['env'])
        cfg = config['agent']
        ns.agent = Agent(env.observation_space, env.action_space, **cfg)
        ns.lock = self.manager().RWLock()

    def evaluate(self):
        # update model
        self.pull_model()
        version = self.manager().eval.meta['version']

        # evaluate
        gear = self.manager().eval.gear
        reports = gear.apply('evaluate', args=())

        # reduce reports
        reduced_report = {k: 0 for k in reports[0].keys()}
        for report in reports:
            for k, v in report.items():
                reduced_report[k] += v / len(reports)

        # send to tb
        ops = []
        for k, v in reduced_report.items():
            ops += Tensorboard.make_numeric_ops(f'eval/{k}', v, version)
        self.remote('tb').rpc_async().add_ops(ops)

    def pull_model(self):
        ns = self.manager().eval
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        with self.manager().eval.lock.wlock():
            ns.agent.load_state_dict(state_dict)
            ns.meta = meta
        
    def close(self):
        self.manager().eval.gear.close()


class SubEvaluator(mp.Target):
    def _start(self):
        self.manager().eval.gear.connect(
            lambda method, *args, **kwargs: getattr(self, method)(*args, **kwargs)
        )

    def evaluate(self):
        # agent
        agent = self.manager().eval.agent

        # env
        config = self.kwargs()['config']
        env = gym.make(**config['env'])

        # reset env
        obs, _ = env.reset()
        terminated = False
        report = {'reward': 0, 'steps': 0}
        while not terminated:
            # step
            action = agent.act({
                'observation': obs.unsqueeze(0)
            }).squeeze()
            obs, rwd, terminated, truncated, info = env.step(action)
            report['reward'] += rwd
            report['steps'] += 1

        return report