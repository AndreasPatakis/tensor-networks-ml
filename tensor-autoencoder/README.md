
# Tensor Train Autoencoder

## Overview

Autoencoder implementation using Tensor Train decomposition for efficient dimensionality reduction.

## Key Features

- TT decomposition for encoder/decoder
- Adjustable TT-rank
- Image reconstruction visualization

## Usage

```python
model = TensorTrainAE(input_shape=(8,8), rank=4)

# Training
optimizer = optim.Adam(model.parameters())
for epoch in range(100):
    reconstructions = model(images)
    loss = F.mse_loss(reconstructions, images)
    loss.backward()
    optimizer.step()
