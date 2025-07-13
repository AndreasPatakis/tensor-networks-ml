# bayesian_tn.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BayesianTensorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=4, n_samples=10):
        super(BayesianTensorNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.n_samples = n_samples
        
        # Mean and log variance parameters for each tensor
        self.core_means = nn.ParameterList([
            nn.Parameter(torch.randn(bond_dim, 2, bond_dim))
            for _ in range(input_dim)
        ])
        self.core_logvars = nn.ParameterList([
            nn.Parameter(torch.zeros(bond_dim, 2, bond_dim))
            for _ in range(input_dim)
        ])
        
        # Boundary parameters
        self.left_mean = nn.Parameter(torch.randn(1, bond_dim))
        self.left_logvar = nn.Parameter(torch.zeros(1, bond_dim))
        self.right_mean = nn.Parameter(torch.randn(bond_dim, output_dim))
        self.right_logvar = nn.Parameter(torch.zeros(bond_dim, output_dim))
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x):
        # Monte Carlo samples
        logits = []
        
        for _ in range(self.n_samples):
            # Sample cores
            cores = []
            for i in range(self.input_dim):
                core = self.reparameterize(self.core_means[i], self.core_logvars[i])
                cores.append(core)
            
            # Sample boundaries
            left = self.reparameterize(self.left_mean, self.left_logvar)
            right = self.reparameterize(self.right_mean, self.right_logvar)
            
            # Contract the MPS
            x_bin = (x > 0.5).float()
            left_contracted = left
            for i in range(self.input_dim):
                core_slice = cores[i][:, int(x_bin[0,i]), :]
                left_contracted = torch.matmul(left_contracted, core_slice)
            
            output = torch.matmul(left_contracted, right)
            logits.append(output)
        
        logits = torch.stack(logits)
        return logits

# Generate data
X, y = make_moons(n_samples=1000, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize model
model = BayesianTensorNetwork(input_dim=2, output_dim=2, bond_dim=4, n_samples=5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass with multiple samples
    logits = model(X_train[0].unsqueeze(0))  # Using first sample for demo
    
    # Compute loss for each sample and average
    losses = [F.cross_entropy(logits[i], y_train[0].unsqueeze(0)) 
              for i in range(model.n_samples)]
    loss = torch.mean(torch.stack(losses))
    
    # Add KL divergence regularization
    kl_div = 0
    for mean, logvar in zip(model.core_means, model.core_logvars):
        kl_div += -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_div += -0.5 * torch.sum(1 + model.left_logvar - model.left_mean.pow(2) - model.left_logvar.exp())
    kl_div += -0.5 * torch.sum(1 + model.right_logvar - model.right_mean.pow(2) - model.right_logvar.exp())
    
    loss += 0.001 * kl_div
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Uncertainty visualization
def plot_uncertainty(model, X, y):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-2, 2, 50))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    with torch.no_grad():
        samples = model(grid)
        probs = F.softmax(samples, dim=-1)
        mean_probs = probs.mean(0)
        std_probs = probs.std(0)
    
    plt.figure(figsize=(15, 5))
    
    # Mean prediction
    plt.subplot(1, 3, 1)
    plt.contourf(xx, yy, mean_probs[:, 1].reshape(xx.shape), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Mean Prediction")
    
    # Uncertainty
    plt.subplot(1, 3, 2)
    uncertainty = std_probs.mean(-1)  # Average std over classes
    plt.contourf(xx, yy, uncertainty.reshape(xx.shape), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Uncertainty")
    
    # Sample predictions
    plt.subplot(1, 3, 3)
    sample_pred = torch.argmax(probs[0], dim=-1)
    plt.contourf(xx, yy, sample_pred.reshape(xx.shape), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Sample Prediction")
    
    plt.tight_layout()
    plt.show()

plot_uncertainty(model, X_test.numpy(), y_test.numpy())