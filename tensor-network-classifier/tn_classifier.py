# tn_classifier.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class TensorNetworkClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=4):
        super(TensorNetworkClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        
        # Feature map tensors
        self.feature_maps = nn.ParameterList([
            nn.Parameter(torch.randn(2, bond_dim))  # Binary feature maps
            for _ in range(input_dim)
        ])
        
        # Central tensor
        self.central_tensor = nn.Parameter(torch.randn(bond_dim**input_dim, output_dim))
        
    def forward(self, x):
        # Convert input to binary features (for simplicity)
        x_bin = (x > 0).float()
        
        # Apply feature maps
        mapped_features = []
        for i in range(self.input_dim):
            mapped = torch.matmul(x_bin[:, i:i+1], self.feature_maps[i])
            mapped_features.append(mapped)
        
        # Compute tensor product of all mapped features
        contracted = mapped_features[0]
        for i in range(1, self.input_dim):
            contracted = torch.einsum('bi,bj->bij', contracted, mapped_features[i])
            contracted = contracted.reshape(-1, self.bond_dim**(i+1))
        
        # Final contraction with central tensor
        output = torch.matmul(contracted, self.central_tensor)
        return output

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize model
model = TensorNetworkClassifier(input_dim=10, output_dim=3, bond_dim=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 50
batch_size = 32
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            test_output = model(X_test)
            predicted = torch.argmax(test_output, dim=1)
            accuracy = (predicted == y_test).float().mean()
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}')