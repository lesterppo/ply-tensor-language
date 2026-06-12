"""
train_v2.py — Complete neural network training using Ply v2.0 API.
Demonstrates: param(), Adam optimizer, step(), save()/load().
Model: 2-layer MLP fitting y = sin(x) + noise.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tensor import Tensor, param, tensor_relu
from optim import Adam

# Generate data
np.random.seed(42)
N = 200
X_data = np.random.randn(N, 1).astype(np.float32) * 2
Y_data = (np.sin(X_data) + 0.1 * np.random.randn(N, 1)).astype(np.float32)

X = Tensor(X_data)
Y = Tensor(Y_data)

# Create parameters using param() — automatically tracked
hidden_dim = 16
W1 = param(np.random.randn(1, hidden_dim).astype(np.float32) * np.sqrt(2.0/1), name='W1')
b1 = param(np.zeros(hidden_dim, dtype=np.float32), name='b1')
W2 = param(np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0/hidden_dim), name='W2')
b2 = param(np.zeros(1, dtype=np.float32), name='b2')

params = [W1, b1, W2, b2]
opt = Adam(params, lr=0.01)

for epoch in range(500):
    # Forward
    h = X @ W1 + b1
    a = tensor_relu(h)
    pred = a @ W2 + b2
    diff = pred - Y
    loss = (diff * diff).mean()

    # Optimizer step (zero_grad + backward + update in one call)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        ss_res = ((pred.data - Y_data) ** 2).sum()
        ss_tot = ((Y_data - Y_data.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f'Epoch {epoch:3d}: loss={loss.data.item():.6f}, R²={r2:.4f}')

print(f'Final loss: {loss.data.item():.6f}')

# Save model
np.savez('/tmp/ply_model.npz',
         W1=W1.data, b1=b1.data,
         W2=W2.data, b2=b2.data)
print('Saved to /tmp/ply_model.npz')

# Load and verify
loaded = np.load('/tmp/ply_model.npz')
print(f'Loaded W1 shape: {loaded["W1"].shape}')

# Compute final predictions
pred_final = (X @ W1 + b1)
pred_final = tensor_relu(pred_final)
pred_final = pred_final @ W2 + b2
ss_res = ((pred_final.data - Y_data) ** 2).sum()
ss_tot = ((Y_data - Y_data.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
print(f'Final R²: {r2:.4f}')
