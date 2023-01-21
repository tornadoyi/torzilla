import gym

class CartPoleWrapper(gym.Wrapper):
    
    def step(self, *args, **kwargs):
        rets = list(super().step(*args, **kwargs))
        
        # reward shaping
        obs = rets[0]
        x, x_dot, theta, theta_dot = obs
        r1 = (self.x_threshold - abs(x)) / self.x_threshold - 0.8
        r2 = (self.theta_threshold_radians - abs(theta)) / self.theta_threshold_radians - 0.5
        rets[1] = r1 + r2
        
        return tuple(rets)