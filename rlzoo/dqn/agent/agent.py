import torch
from torzilla import nn
from torzilla.nn import functional as F
from .model import QFunction

class Agent(nn.Module):
    def __init__(
        self,
        ob_space, 
        ac_space,
        gamma=0.9,
        double_q=False,
        eps=0.1,
        eps_annealing=0,
        q_func_args={},
    ):
        super().__init__()
        self.gamma = gamma
        self.double_q = double_q
        self.num_action = ac_space.n
        self.q = QFunction(ob_space, ac_space, **q_func_args)
        self.qt = QFunction(ob_space, ac_space, **q_func_args)
        self.eps = nn.parameter.Parameter(torch.tensor(eps), requires_grad=False)
        self.eps_annealing = eps_annealing

    def act(self, inputs, eps=None):
        with torch.no_grad():
            return self._act(inputs, eps)
            
    def _act(self, inputs, eps_greedy=True):
        obs = inputs['observation']

        # deterministic actions
        q_values = self.q(obs)
        deterministic_actions = torch.argmax(q_values, axis=1)
        
        # epsilon greedy
        N = q_values.shape[0]
        rand_actions = torch.randint(low=0, high=self.num_action, size=(N, 1))
        if eps_greedy:
            need_rand = torch.rand_like(rand_actions) < self.eps
            self.eps = torch.clip(self.eps - self.eps_annealing, 0, 1)
        else:
            need_rand = torch.zeros_like(rand_actions).to(torch.bool)
        stochastic_actions = torch.where(need_rand, rand_actions, deterministic_actions)

        return stochastic_actions

    def learn(self, inputs):
        obs, next_obs = inputs['observation'], inputs['next_observation']
        rew, done =  inputs['reward'], inputs['done']
        act = inputs['action']

        # q eval
        q_values = self.q(obs)
        q_act_value = torch.sum(q_values * F.one_hot(act, self.num_action), axis=1)

        # target q 
        qt_values = self.qt(next_obs)

        # double q
        if self.double_q:
            q_next_values = self.q(next_obs)
            q_next_act = torch.argmax(q_next_values, axis=1)
            qt_act_value = torch.sum(qt_values * F.one_hot(q_next_act, self.num_action), axis=1)
        else:
            qt_act_value = torch.max(qt_values, axis=1)

        qt_mask_value = (1.0 - done) * qt_act_value

        # bellman equation
        td_target = rew + self.gamma * qt_mask_value
        loss = F.huber_loss(q_act_value, td_target.detach())
        return loss