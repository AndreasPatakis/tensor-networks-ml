
# Matrix Product States for Classification

## Overview

Implementation of a Matrix Product State (MPS) classifier for binary classification tasks. Demonstrates how quantum-inspired tensor network architectures can be applied to classical machine learning problems.

## Key Features

- MPS architecture with adjustable bond dimension
- Binary feature mapping
- End-to-end differentiable training

## Usage

```python
from mps_classification import MPSClassifier

model = MPSClassifier(input_dim=64, output_dim=2, bond_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
