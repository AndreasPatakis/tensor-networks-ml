
# Bayesian Tensor Networks

## Overview

Implementation of Bayesian tensor networks that provide uncertainty estimates alongside predictions.

## Key Features

- Variational inference for tensor networks
- Monte Carlo sampling during forward pass
- Uncertainty visualization

## Usage

```python
model = BayesianTensorNetwork(input_dim=2, output_dim=2, n_samples=5)

# Get predictions with uncertainty
logits = model(inputs)  # Returns [n_samples, batch_size, output_dim]
probs = torch.softmax(logits, dim=-1)
mean_probs = probs.mean(dim=0)
std_probs = probs.std(dim=0)
