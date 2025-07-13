
# Quantum-Inspired Reinforcement Learning

## Overview

Tensor network implementation of Q-learning for the CartPole environment, demonstrating how tensor networks can represent value functions in RL.

## Key Features

- Tensor network Q-function approximation
- Experience replay buffer
- ε-greedy exploration

## Usage

```python
env = gym.make('CartPole-v1')
q_network = TensorNetworkQFunction(state_dim=4, action_dim=2)

# Training
for episode in range(200):
    state = env.reset()
    while True:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        # ... training steps ...
