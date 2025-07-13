# Tensor Networks for Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains implementations of tensor network methods applied to machine learning tasks. Tensor networks provide efficient, interpretable, and scalable approaches to modern ML challenges.

## Projects

1. **Matrix Product States for Classification** - `1-matrix-product-states/`
   - Binary classification using MPS architecture

2. **Tensor Train Decomposition** - `2-tensor-train-decomposition/`
   - Data compression with TT decomposition

3. **Tensor Network Classifier** - `3-tensor-network-classifier/`
   - General classification with tensor networks

4. **Quantum-Inspired RL** - `4-quantum-inspired-rl/`
   - Reinforcement learning with tensor network Q-functions

5. **Bayesian Tensor Networks** - `5-uncertainty-quantification/`
   - Uncertainty estimation in tensor networks

6. **Tensor Network Autoencoder** - `6-tensor-autoencoder/`
   - Dimensionality reduction with tensor trains

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Scikit-learn
- Matplotlib (for visualization)
- Gym (for RL environment)

## Installation

```bash
git clone https://github.com/andreaspatakis/tensor-networks-ml.git
cd tensor-networks-ml
pip install -r requirements.txt
