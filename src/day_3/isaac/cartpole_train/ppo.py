import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

class PPO:
    def __init__(self, env) -> None:
        self.env = env
        self.num_inputs  = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.shape[0]
        
        #Hyper params:
        self.hidden_size      = 256
        self.lr               = 3e-4
        self.num_steps        = 20
        self.mini_batch_size  = 5
        self.ppo_epochs       = 4
        self.threshold_reward = 200
        
        self.model = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def test_env(self,vis=False):
        state = self.env.reset()
        if vis: self.env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = self.env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis: self.env.render()
            total_reward += reward
        return total_reward

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
            
            

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def train(self, max_frames = 150000) -> None:
        state = self.env.reset()
        early_stop = False
        test_rewards = []
        frame_idx  = 0

        while frame_idx < max_frames and not early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0
            
            

            for _ in range(self.num_steps):
                state = torch.FloatTensor(state).to(device)
                dist, value = self.model(state)

                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                #print(action, state, next_state, reward, done)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                # FloatTensor expects an array
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1
                
                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env() for _ in range(10)])
                    test_rewards.append(test_reward)
                    #print(frame_idx, test_rewards)
                    if test_reward > self.threshold_reward: early_stop = True
                    
            print(frame_idx, test_rewards)
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            
            self.ppo_update(self.ppo_epochs, self.mini_batch_size, states, actions, log_probs, returns, advantage)
