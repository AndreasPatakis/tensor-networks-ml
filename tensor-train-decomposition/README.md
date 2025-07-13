# Tensor Train Decomposition for Data Compression

## Overview

Implementation of Tensor Train (TT) decomposition for compressing high-dimensional data with controllable accuracy.

## Key Features

- TT decomposition algorithm
- Reconstruction from TT cores
- Visualization of compression effects

## Usage

```python
from tt_compression import tensor_train_decomposition, reconstruct_from_tt

# Decompose 8x8 image with rank 4
cores = tensor_train_decomposition(image, ranks=[4,4])
reconstructed = reconstruct_from_tt(cores)

# Calculate compression ratio
original_size = image.size
compressed_size = sum(core.size for core in cores)
