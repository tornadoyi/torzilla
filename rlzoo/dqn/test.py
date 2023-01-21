"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
from rlzoo.zoo import gym
from torzilla.distributed.optim import ReducedOptimizer
from rlzoo.dqn.agent import Agent

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
# env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# agent
agent = Agent(
    env.observation_space, 
    env.action_space,
    gamma = GAMMA,
    eps = (1 - EPSILON),
    eps_annealing = 2,
    qtarget_update_freq = TARGET_REPLACE_ITER,
)

# optimizer
optimizer = torch.optim.Adam(
    [p for p in agent.parameters() if p.requires_grad], 
    lr=LR
)

# rb
M = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
memory_counter = 0

def store_transition(s, a, r, s_):
    global memory_counter, M
    transition = np.hstack((s, [a, r], s_))
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    M[index, :] = transition
    memory_counter += 1

def sample():
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = M[sample_index, :]
    b_s = torch.FloatTensor(b_memory[:, :N_STATES])
    b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).squeeze()
    b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).squeeze()
    b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
    b_d = torch.zeros_like(b_a, dtype=torch.bool)
    return {'observation': b_s, 'action': b_a, 'reward': b_r, 'next_observation': b_s_, 'done': b_d}

print('\nCollecting experience...')
for i_episode in range(400):
    s, _ = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = agent.act({'observation': torch.as_tensor(s).unsqueeze(0)}).squeeze().numpy()

        # take action
        s_, r, done, trunc, info = env.step(torch.as_tensor(a))

        
        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        store_transition(s, a, r, s_)

        ep_r += r.numpy()
        if memory_counter > MEMORY_CAPACITY:
            inputs = sample()
            res = agent.learn(inputs)
            loss = res['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_