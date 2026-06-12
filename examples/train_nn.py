#!/usr/bin/env python3
"""
train_nn.py — Train a 2-layer neural network using Ply gradients.
Model: X -> Linear(1->16) -> ReLU -> Linear(16->1)
Loss: MSE
"""
import sys
sys.path.insert(0, '/home/peter/ply')
import numpy as np
from tensor import Tensor, tensor_relu

# Generate synthetic nonlinear data: y = sin(x) + noise
np.random.seed(42)
N = 200
X_data = np.random.randn(N, 1).astype(np.float32) * 2
Y_data = (np.sin(X_data) + 0.1 * np.random.randn(N, 1)).astype(np.float32)

X = Tensor(X_data)
Y = Tensor(Y_data)

# Initialize weights (He init)
hidden_dim = 16
W1 = Tensor(np.random.randn(1, hidden_dim).astype(np.float32) * np.sqrt(2.0 / 1))
b1 = Tensor(np.zeros(hidden_dim, dtype=np.float32))
W2 = Tensor(np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim))
b2 = Tensor(np.zeros(1, dtype=np.float32))

lr = 0.01
for epoch in range(500):
    # Forward pass
    h = X @ W1 + b1
    a = tensor_relu(h)
    pred = a @ W2 + b2
    diff = pred - Y
    loss = (diff * diff).mean()

    # Backward
    for p in [W1, b1, W2, b2]:
        p.grad = None
    loss.backward()

    # Gradient clipping
    for p in [W1, b1, W2, b2]:
        p.grad = np.clip(p.grad, -1.0, 1.0)

    # Update
    W1.data -= lr * W1.grad
    b1.data -= lr * b1.grad
    W2.data -= lr * W2.grad
    b2.data -= lr * b2.grad

    if epoch % 100 == 0:
        # Compute R²-ish metric
        ss_res = ((pred.data - Y_data) ** 2).sum()
        ss_tot = ((Y_data - Y_data.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f'Epoch {epoch:3d}: loss={loss.data.item():.6f}, R²={r2:.4f}')

print(f'Final loss: {loss.data.item():.6f}')
