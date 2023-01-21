import gym
import torch

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