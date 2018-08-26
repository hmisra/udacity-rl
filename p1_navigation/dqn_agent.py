import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    'Interacts with and learns from the environment'
    
    def __init__(self, state_size, action_size, seed, num_layers, hidden_units):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # local network is used to implement the current policy 
        self.qnetwork_local = QNetwork(state_size, action_size, seed, num_layers, hidden_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, amsgrad=True)
        
        # target network is used to estimate the Q-value function 
        # params are copied from local network every UPDATE_EVERY steps
        self.qnetwork_target = QNetwork(state_size, action_size, seed, num_layers, hidden_units).to(device)
        
        # Memory containing experiences to be used during experience replay
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
        
    def step(self, state, action, reward, next_state, done):
        # add the experience into memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                # once there are sufficient experience samples, sample and learn
                experiences = self.memory.sample()
                self.learn_DDQN(experiences, GAMMA)
                
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy"
           Epsilon-greedy algorithm is used to select the actions.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
        
    def learn_vanilla_DQN(self, experiences, gamma):
        """Uses DQN (current policy is calculated via local network but the 
           learning targets are calculated using a separate target network).
        """
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) 
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)               # try using Huber loss
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network by copying weights from local network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        
    def learn_DDQN(self, experiences, gamma):
        """Uses double-DQN whereby actions are chosen using current policy but the
           actions are evaluated on target network to prevent overestimations of Q-values.
        """
        states, actions, rewards, next_states, dones = experiences
        # Double DQN
        # (1) use current policy (local network) to select best actions from next states
        # (2) get predicted Q-values (target network) for best actions selected by current policy    
        Q_local_argmax = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) 
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_local_argmax) 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) 
        Q_expected = self.qnetwork_local(states).gather(1, actions) 
        loss = F.smooth_l1_loss(Q_expected, Q_targets)              # try using Huber loss
        # Backprop 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Copy weights from local network to target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft/smooth updating of target model's parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer():
    'Fixed size buffer to store experience tuples'
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
    
    
    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    
    def sample(self):
        "Randomly sample a batch of experiences from memory."
        exps = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones
    
    
    def __len__(self):
        return len(self.memory)
    