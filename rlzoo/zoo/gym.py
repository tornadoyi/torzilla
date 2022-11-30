import gym
from gym.wrappers import *
import torch

__ENV_CTRL__ = [
    'acrobot',
    'cartpole',
    'mountaincarcontinuous',
    'mountaincar',
    'pendulum',
]

__ENV_ATARI__ = [
    'breakout',
]

def make(id):
    env = gym.make(id)
    return _wrap(env)

def _wrap(env):
    name = env.spec.id.lower()
    for n in __ENV_CTRL__:
        if not name.startswith(n): continue
        env = _classic_control(env)
        break

    for n in __ENV_ATARI__:
        if not name.startswith(n): continue
        env = _atri(env)
        break

    return TensorWrapper(env)

def _classic_control(env):
    return env


def _atri(env):
    return AtariPreprocessing(env)


class TensorWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return torch.as_tensor(obs), info

    def step(self, action):
        action = action.numpy()
        rets = super().step(action)

        # convert obs, rew, term, tunck to tensor
        rets = list(rets)
        for i in range(len(rets)-1):      
            rets[i] = torch.as_tensor(rets[i])

        return tuple(rets)