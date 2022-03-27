import gym
import warnings
from DQN import *
from Agent import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
env = gym.make('MountainCar-v0')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

if __name__ == '__main__':
    layers = (
        nn.Linear(n_states, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions),
    )

    Model = DQN(layers, lr=0.0001, optim_method=optim.Adam)

    MountainCarAgent = Agent(env, Model, n_actions, goal=-110, min_score=-200,
                             eps_start=1, eps_end=0.001, eps_decay=0.9, gamma=0.99,
                             batch_size=64, memory_size=100000, max_episode=2000)

    scores = MountainCarAgent.train()
    MountainCarAgent.test(episodes=100)
    episodes = range(len(scores))
    plt.plot(episodes, scores)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
