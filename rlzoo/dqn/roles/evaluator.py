from rlzoo.zoo import gym
from rlzoo.zoo.role import Role
from ..agent import Agent


class Evaluator(Role):
    
    def _start(self):
        config = self.kwargs()['config']
        self.env = gym.make(**config['env'])
        cfg = config['agent']
        self.agent = Agent(self.env.observation_space, self.env.action_space, **cfg)

    def evaluate(self, max_step=-1):
        # pull model
        meta = self.pull_model()
        self.last_version = meta['version']

        # reset env
        obs, _ = self.env.reset()
        terminated = False
        stats = {'reward': 0, 'steps': 0}
        while not terminated:
            # step
            action = self.agent.act({
                'observation': obs.unsqueeze(0)
            }).squeeze()
            obs, rwd, terminated, truncated, info = self.env.step(action)
            stats['reward'] += rwd
            stats['steps'] += 1

            # max step break
            if max_step >= 0 and stats['steps'] >= max_step:
                break

        return stats

    def pull_model(self):
        state_dict, meta = self.remote('ps').rpc_sync().get(meta=True)
        self.agent.load_state_dict(state_dict)
        return meta
            

