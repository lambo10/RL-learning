import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    """Deep Q-Network Mapping States to Action Values."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience Replay Buffer for DQN Training."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32), 
                np.array(action, dtype=np.int64), 
                np.array(reward, dtype=np.float32), 
                np.array(next_state, dtype=np.float32), 
                np.array(done, dtype=np.uint8))
                
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent handling training and inference."""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64
        
        # Policy and Target Networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=20000)
        self.steps_done = 0
        
    def act(self, state, eval_mode=False):
        """Select action using Epsilon-Greedy strategy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()
            
    def cache(self, state, action, reward, next_state, done):
        """Cache transition in experience replay memory."""
        self.memory.push(state, action, reward, next_state, done)
        
    def learn(self):
        """Train the policy network on a batch of replay experiences."""
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch Tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)
        
        # Compute Current Q-values
        curr_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        
        # Compute Target Q-values using Target Network (DDQN/DQN style)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
            
        # Huber Loss for training stability
        loss = nn.SmoothL1Loss()(curr_q, target_q)
        
        # Optimize Policy Net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps_done += 1
        return loss.item()
        
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filepath):
        """Save network weights."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.policy_net.eval()
