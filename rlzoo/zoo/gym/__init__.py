import gymnasium as gym
from .wrappers import TensorWrapper, AtariPreprocessing, envs as env_wrappers


__ENV_CLASS__ = {
    'ctrl': [
        'CartPole',
        'Acrobot',
        'MountainCarContinuous',
        'MountainCar',
        'Pendulum',
    ],

    'atari': [
        'Breakout',
    ]
}


__ENVS__ = {}
for cls, env_names in __ENV_CLASS__.items():
    for name in env_names:
        env_wrap = getattr(env_wrappers, f'{name}Wrapper', None)
        __ENVS__[name] = (cls, env_wrap)


def make(
    id, 
    enable_tensor_wrapper=True,
    enable_id_wrapper=False,
    **kwargs
):
    env = gym.make(id, **kwargs)
    name = env.spec.id.split('-')[0]
    
    info = __ENVS__.get(name, None)
    if info is not None:
        cls, env_wrap = info
        if cls == 'atari':
            env = AtariPreprocessing(env)
        if enable_id_wrapper and env_wrap is not None:
            env = env_wrap(env)

    if enable_tensor_wrapper:
        env = TensorWrapper(env)

    return env

    