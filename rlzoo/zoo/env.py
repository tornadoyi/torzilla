import gym
from gym.wrappers import *

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

def make_env(id):
    env = gym.make(id)
    return _wrap(env)

def _wrap(env):
    name = env.spec.id.lower()
    for n in __ENV_CTRL__:
        if not name.startswith(n): continue
        return _classic_control(env)

    for n in __ENV_ATARI__:
        if not name.startswith(n): continue
        return _atri(env)

    return env

def _classic_control(env):
    return env


def _atri(env):
    return AtariPreprocessing(env)