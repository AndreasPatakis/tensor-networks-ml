# mps_classification.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=4):
        super(MPSClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        
        # Create tensors for each site
        self.tensors = nn.ParameterList([
            nn.Parameter(torch.randn(bond_dim, 2, bond_dim) 
            for _ in range(input_dim)
        ])
        
        # Boundary vectors
        self.left_boundary = nn.Parameter(torch.randn(1, bond_dim))
        self.right_boundary = nn.Parameter(torch.randn(bond_dim, output_dim))
        
    def forward(self, x):
        # Convert input to binary features (for simplicity)
        x_bin = (x > 0.5).float()
        
        # Contract the MPS
        left = self.left_boundary
        for i in range(self.input_dim):
            # Select the appropriate tensor slice based on input
            tensor = self.tensors[i][:, int(x_bin[0,i]), :]
            left = torch.matmul(left, tensor)
        
        # Final contraction with right boundary
        output = torch.matmul(left, self.right_boundary)
        return output

# Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target

# Binary classification: even vs odd digits
y = (y % 2 == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize model
model = MPSClassifier(input_dim=64, output_dim=2, bond_dim=8)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train[0].unsqueeze(0))  # Just using first sample for demo
    loss = criterion(outputs, y_train[0].unsqueeze(0))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation
with torch.no_grad():
    test_output = model(X_test[0].unsqueeze(0))
    predicted = torch.argmax(test_output)
    print(f'Predicted: {predicted.item()}, Actual: {y_test[0].item()}')