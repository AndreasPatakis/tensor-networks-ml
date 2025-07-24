import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=8):
        super(MPSClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.local_dim = 2  # local feature dimension

        # Initialize MPS tensors: list of tensors with shape (bond_dim_left, local_dim, bond_dim_right)
        self.tensors = nn.ParameterList()
        for i in range(input_dim):
            left_dim = bond_dim if i > 0 else 1
            right_dim = bond_dim if i < input_dim - 1 else 1
            tensor = nn.Parameter(torch.randn(left_dim, self.local_dim, right_dim) * 0.1)
            self.tensors.append(tensor)

        # Linear layer mapping from the final scalar to class logits
        self.classifier = nn.Linear(1, output_dim)

    def feature_map(self, x):
        # x shape: [batch_size=1, input_dim]
        # Map each scalar x_i to 2D vector phi(x_i)
        # Normalize x to [0,1] first
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)  # shape: [1, input_dim]

        # Apply feature map phi(x_i) = [cos(pi/2 x_i), sin(pi/2 x_i)]
        phi_0 = torch.cos(np.pi / 2 * x_norm)  # shape: [1, input_dim]
        phi_1 = torch.sin(np.pi / 2 * x_norm)
        phi = torch.stack([phi_0, phi_1], dim=2)  # shape: [1, input_dim, 2]

        return phi  # shape: [1, input_dim, 2]

    def forward(self, x):
        # x shape: [1, input_dim]
        batch_size = x.shape[0]
        phi = self.feature_map(x)  # [1, input_dim, 2]

        # Start contraction with left boundary vector (scalar 1)
        left = torch.ones(batch_size, 1, 1, device=x.device)  # shape: [batch, 1, bond_dim_left]

        for i in range(self.input_dim):
            # Tensor shape: (bond_dim_left, local_dim, bond_dim_right)
            M = self.tensors[i]  # shape: [D_left, d, D_right]
            M = M.unsqueeze(0)   # [1, D_left, d, D_right]
            
            # Select feature vector at site i: shape [batch, d]
            phi_i = phi[:, i, :].unsqueeze(1)  # [batch, 1, d]

            # Contract: left [batch, 1, D_left], M [1, D_left, d, D_right], phi_i [batch, 1, d]
            # We want to contract over D_left and d dims:
            # einsum: left (b,1,Dl), M (1,Dl,d,Dr), phi_i (b,1,d)
            # -> result shape: (batch, 1, D_right)
            left = torch.einsum('bsl,sldr,bsd->bsr', left, M, phi_i)
            # left shape after einsum: (batch, 1, D_right)

        # After all contractions left shape: [batch, 1, 1], squeeze to scalar
        final_scalar = left.squeeze(-1).squeeze(-1)  # shape: [batch]

        # Classify
        logits = self.classifier(final_scalar.unsqueeze(-1))  # [batch, output_dim]

        return logits

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

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model, loss, optimizer
model = MPSClassifier(input_dim=64, output_dim=2, bond_dim=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(len(X_train)):
        optimizer.zero_grad()
        outputs = model(X_train[i].unsqueeze(0))
        loss = criterion(outputs, y_train[i].unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(X_train):.4f}")

# Evaluation
model.eval()
correct = 0
total = len(X_test)
with torch.no_grad():
    for i in range(total):
        outputs = model(X_test[i].unsqueeze(0))
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == y_test[i]).item()
accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
