import torch
import random
import numpy as np
from ReplayMemory import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent:
    def __init__(self, env, Model, n_actions, goal, min_score,
                 eps_start=1, eps_end=0.001, eps_decay=0.9, gamma=0.99,
                 batch_size=64, memory_size=100000, max_episode=2000, upd_rate=1):
        self.env = env
        self.n_actions = n_actions  # number of possible actions
        self.goal = goal  # the score to reach during learning
        self.min_score = min_score  # min score to complete the episode
        self.eps_start = eps_start
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_rate = upd_rate  # how often we update our target network
        self.Model = Model  # DQN instance
        self.max_episode = max_episode  # how long we train our agent
        self.memory = ReplayMemory(memory_size)  # Replay buffer
        #self.Model.to(device)?

    def act(self, state, eps):  # epsilon greedy policy
        if random.random() < eps:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                result = self.Model.Q_estimate(state).max(1)[1]
                return result.view(1, 1)

    def optimize(self):  # experience replay
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        estimate_value = self.Model.Q_estimate(
            state_batch).gather(1, action_batch)

        Q_value_next = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            Q_value_next[~done_batch] = self.Model.Q_target(next_state_batch).max(1)[
                0].detach()[~done_batch]
        target_value = (Q_value_next * self.gamma) + reward_batch

        self.Model.update_parameters(estimate_value, target_value)

    def train(self):  # learning procedure
        all_scores = []
        successful_sequences = 0
        for ep in range(1, self.max_episode + 1):
            state = self.env.reset()
            state = torch.tensor(state).to(device).float().unsqueeze(0)
            done = False
            episode_reward = 0

            while not done:
                action = self.act(state, self.eps)
                action = torch.tensor(action).to(device)

                next_state, reward, done, info = self.env.step(action.item())
                episode_reward += reward

                modified_reward = reward + 300 * \
                    (self.gamma * abs(next_state[1]) - abs(state[0][1]))

                next_state = torch.tensor(next_state).to(
                    device).float().unsqueeze(0)
                modified_reward = torch.tensor(modified_reward).to(
                    device).float().unsqueeze(0)
                done = torch.tensor(done).to(device).unsqueeze(0)

                self.memory.push(state, action, next_state,
                                 modified_reward, done)
                state = next_state

                self.optimize()  # experience replay

            if ep % self.target_update_rate == 0:
                self.Model.update_target()

            self.eps = max(self.eps_end, self.eps * self.eps_decay)
            all_scores.append(episode_reward)

            if ep % 100 == 0:
                print('episode', ep, ':', np.mean(
                    all_scores[:-100:-1]), 'average score')

            if np.mean(all_scores[:-100:-1]) >= self.goal:
                successful_sequences += 1
                if successful_sequences == 5:
                    print('success at episode', ep)
                    return all_scores
            else:
                successful_sequences = 0

        return all_scores

    def test(self, episodes=50, render=False):  # test trained agent
        state = self.env.reset()
        state = torch.tensor(state).to(device).float().unsqueeze(0)
        ep_count = 0
        current_episode_reward = 0
        scores = []
        while ep_count < episodes:
            if render:
                self.env.render()
            action = self.act(state, 0)
            state, reward, done, _ = self.env.step(action.item())
            state = torch.tensor(state).to(device).float().unsqueeze(0)
            current_episode_reward += reward

            if done:
                ep_count += 1
                scores.append(current_episode_reward)
                current_episode_reward = 0
                state = self.env.reset()
                state = torch.tensor(state).to(device).float().unsqueeze(0)

        print('average score:', sum(scores) / len(scores))
        print('max reward:', max(scores))
        print('-----')
        print()

    def save(self, name='agent.pkl'):  # save policy network
        self.Model.save(name)
