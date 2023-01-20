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
        eps=0.0,
        eps_annealing=0,
        qtarget_update_freq=1,
        q_func_args={},
    ):
        super().__init__()
        self.gamma = gamma
        self.double_q = double_q
        self.num_action = ac_space.n
        self.q = QFunction(ob_space, ac_space, **q_func_args)
        self.qt = QFunction(ob_space, ac_space, **q_func_args)
        self.eps = eps
        self.eps_annealing = eps_annealing
        self.qtarget_update_freq = qtarget_update_freq

        self.num_learn = nn.parameter.Parameter(torch.tensor(0), requires_grad=False)
        self.num_qt_update = nn.parameter.Parameter(torch.tensor(0), requires_grad=False)

        # set target network with no grad
        for p in self.qt.parameters():
            p.requires_grad = False

    def calc_eps(self, total_step):
        w = 1 - self.num_learn / total_step * self.eps_annealing
        return torch.clip(self.eps * w, 0, 1)

    def act(self, inputs, eps=0):
        with torch.no_grad():
            return self._act(inputs, eps)
            
    def _act(self, inputs, eps):
        obs = inputs['observation']

        # deterministic actions
        q_values = self.q(obs)
        deterministic_actions = torch.argmax(q_values, axis=1)
        
        # epsilon greedy
        N = q_values.shape[0]
        rand_actions = torch.randint(low=0, high=self.num_action, size=(N, ))
        need_rand = torch.rand_like(rand_actions.to(torch.float32)) < eps
        stochastic_actions = torch.where(need_rand, rand_actions, deterministic_actions)

        return stochastic_actions

    def learn(self, inputs):
        # inputs
        obs = inputs['observation']
        next_obs = inputs['next_observation']
        rew = inputs['reward']
        done = inputs['done'].to(torch.float32)
        act = inputs['action']

        # update qtarget
        self.num_learn += 1
        if self.num_learn % self.qtarget_update_freq == 0:
            self.qt.load_state_dict(self.q.state_dict())
            self.num_qt_update += 1

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
            qt_act_value, _ = torch.max(qt_values, axis=1)

        qt_mask_value = (1.0 - done) * qt_act_value

        # bellman equation
        td_target = rew + self.gamma * qt_mask_value
        loss = F.huber_loss(q_act_value, td_target.detach())

        return {
            'loss': loss,
            'qt_value': qt_mask_value,
            'num_qt_update': self.num_qt_update,
        }