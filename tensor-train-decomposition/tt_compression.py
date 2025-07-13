# tt_compression.ipynb

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def tensor_train_decomposition(tensor, ranks):
    """Basic Tensor Train decomposition implementation"""
    shape = tensor.shape
    dim = len(shape)
    cores = []
    
    # Reshape tensor into a matrix for first decomposition
    unfolding = tensor.reshape(shape[0], -1)
    U, S, V = np.linalg.svd(unfolding, full_matrices=False)
    
    # Truncate based on rank
    U = U[:, :ranks[0]]
    S = S[:ranks[0]]
    V = V[:ranks[0], :]
    
    # First core
    core = U.reshape(1, shape[0], ranks[0])
    cores.append(core)
    
    # Reshape V for next decomposition
    current_tensor = (np.diag(S) @ V).reshape(ranks[0]*shape[1], -1)
    
    for i in range(1, dim-1):
        U, S, V = np.linalg.svd(current_tensor, full_matrices=False)
        U = U[:, :ranks[i]]
        S = S[:ranks[i]]
        V = V[:ranks[i], :]
        
        core = U.reshape(ranks[i-1], shape[i], ranks[i])
        cores.append(core)
        current_tensor = (np.diag(S) @ V).reshape(ranks[i]*shape[i+1], -1)
    
    # Last core
    core = current_tensor.reshape(ranks[-1], shape[-1], 1)
    cores.append(core)
    
    return cores

def reconstruct_from_tt(cores):
    """Reconstruct tensor from TT cores"""
    current = cores[0]
    for core in cores[1:]:
        current = np.tensordot(current, core, axes=(-1, 0))
    return np.squeeze(current)

# Load sample data (digit image)
digits = load_digits()
image = digits.images[0]

# Apply TT decomposition
ranks = [4, 4]  # TT ranks
cores = tensor_train_decomposition(image, ranks)

# Reconstruct image
reconstructed = reconstruct_from_tt(cores)

# Calculate compression ratio
original_size = image.size
compressed_size = sum(core.size for core in cores)
compression_ratio = original_size / compressed_size

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Reconstructed (CR: {compression_ratio:.1f}x)")
plt.imshow(reconstructed, cmap='gray')

plt.tight_layout()
plt.show()