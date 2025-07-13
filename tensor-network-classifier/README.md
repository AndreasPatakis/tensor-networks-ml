# Tensor Network Classifier

## Overview

Implementation of a general tensor network classifier for multi-class classification tasks, demonstrating how tensor networks can compete with traditional neural networks.

## Key Features

- Flexible tensor network architecture
- Multi-class classification support
- Adjustable bond dimension (model capacity)
- End-to-end differentiable training

## Usage

```python
from tn_classifier import TensorNetworkClassifier

# Initialize model
model = TensorNetworkClassifier(input_dim=10, output_dim=3, bond_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Evaluation
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
