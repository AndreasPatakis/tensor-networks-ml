# tn_autoencoder.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class TensorTrainAE(nn.Module):
    def __init__(self, input_shape, rank=4):
        super(TensorTrainAE, self).__init__()
        self.input_shape = input_shape
        self.rank = rank
        
        # Encoder TT cores
        self.encoder_cores = nn.ParameterList([
            nn.Parameter(torch.randn(1 if i == 0 else rank, 
                                  input_shape[i], 
                                  rank if i < len(input_shape)-1 else 1))
            for i in range(len(input_shape))
        ])
        
        # Decoder TT cores
        self.decoder_cores = nn.ParameterList([
            nn.Parameter(torch.randn(1 if i == 0 else rank, 
                                  input_shape[i], 
                                  rank if i < len(input_shape)-1 else 1))
            for i in range(len(input_shape))
        ])
        
    def encode(self, x):
        # Reshape input to tensor
        x_tensor = x.view(*self.input_shape)
        
        # Contract with encoder cores
        contracted = x_tensor.unsqueeze(0)  # Add batch dim
        for i in range(len(self.input_shape)):
            core = self.encoder_cores[i]
            contracted = torch.einsum('...i,ijk->...jk', contracted, core)
        
        return contracted.squeeze()
    
    def decode(self, z):
        # Contract with decoder cores
        contracted = z.unsqueeze(-1)  # Add dummy dim
        for i in reversed(range(len(self.input_shape))):
            core = self.decoder_cores[i]
            contracted = torch.einsum('...j,jik->...ik', contracted, core)
        
        return contracted.squeeze()
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

# Load data
digits = load_digits()
X = digits.data / 16.0  # Normalize to [0, 1]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Initialize model
input_shape = (8, 8)  # Digits are 8x8 images
model = TensorTrainAE(input_shape, rank=4)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 100
batch_size = 32
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), batch_size):
        batch_indices = permutation[i:i+batch_size]
        batch_x = X_train[batch_indices]
        
        optimizer.zero_grad()
        reconstructions = model(batch_x)
        loss = criterion(reconstructions, batch_x)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            test_recon = model(X_test[:1])
            test_loss = criterion(test_recon, X_test[:1])
            print(f'Epoch {epoch}, Loss: {test_loss.item()}')

# Visualization
def plot_reconstructions(model, X, n=5):
    with torch.no_grad():
        reconstructions = model(X[:n])
    
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i+1)
        plt.imshow(X[i].view(8, 8), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, n, n+i+1)
        plt.imshow(reconstructions[i].view(8, 8), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_reconstructions(model, X_test)