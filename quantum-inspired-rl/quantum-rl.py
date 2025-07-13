# quantum_inspired_rl.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class TensorNetworkQFunction(nn.Module):
    def __init__(self, state_dim, action_dim, bond_dim=4):
        super(TensorNetworkQFunction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bond_dim = bond_dim
        
        # State feature maps
        self.state_maps = nn.ParameterList([
            nn.Parameter(torch.randn(1, bond_dim)) 
            for _ in range(state_dim)
        ])
        
        # Action feature maps
        self.action_maps = nn.ParameterList([
            nn.Parameter(torch.randn(1, bond_dim))
            for _ in range(action_dim)
        ])
        
        # Central tensor
        self.central_tensor = nn.Parameter(torch.randn(bond_dim, bond_dim, 1))
        
    def forward(self, state, action):
        # State features
        state_features = []
        for i in range(self.state_dim):
            mapped = torch.matmul(state[:, i:i+1], self.state_maps[i])
            state_features.append(mapped)
        
        # Action features (one-hot encoded)
        action_onehot = torch.zeros(action.size(0), self.action_dim)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        action_features = []
        for i in range(self.action_dim):
            mapped = torch.matmul(action_onehot[:, i:i+1], self.action_maps[i])
            action_features.append(mapped)
        
        # Contract state features
        state_contracted = state_features[0]
        for i in range(1, self.state_dim):
            state_contracted = torch.einsum('bi,bj->bij', state_contracted, state_features[i])
            state_contracted = state_contracted.reshape(-1, self.bond_dim)
        
        # Contract action features
        action_contracted = action_features[0]
        for i in range(1, self.action_dim):
            action_contracted = torch.einsum('bi,bj->bij', action_contracted, action_features[i])
            action_contracted = action_contracted.reshape(-1, self.bond_dim)
        
        # Final contraction
        q_value = torch.einsum('bi,bj,ijk->bk', 
                              state_contracted, 
                              action_contracted, 
                              self.central_tensor)
        return q_value.squeeze()

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize Q-network
q_network = TensorNetworkQFunction(state_dim, action_dim, bond_dim=4)
target_network = TensorNetworkQFunction(state_dim, action_dim, bond_dim=4)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 32
replay_buffer = []
buffer_capacity = 10000
update_target_every = 10

def select_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = torch.stack([
                q_network(state_tensor, torch.tensor([a]))
                for a in range(action_dim)
            ])
            return torch.argmax(q_values).item()

# Training loop
episodes = 200
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > buffer_capacity:
            replay_buffer.pop(0)
        
        # Train on batch
        if len(replay_buffer) >= batch_size:
            batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
            states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in batch])
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = torch.stack([
                    target_network(next_states, torch.tensor([a] * batch_size))
                    for a in range(action_dim)
                ])
                max_next_q = torch.max(next_q_values, dim=0)[0]
                targets = rewards + gamma * max_next_q * (1 - dones)
            
            # Compute current Q-values
            current_q = q_network(states, actions)
            
            # Compute loss and update
            loss = nn.MSELoss()(current_q, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        if done:
            break
    
    # Update target network
    if episode % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")