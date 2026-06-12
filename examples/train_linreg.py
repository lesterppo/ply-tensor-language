#!/usr/bin/env python3
"""
train_linreg.py — Train linear regression using Ply gradients.
Demonstrates: forward pass in Ply → gradient in Ply → update in Python.
"""
import sys
sys.path.insert(0, '/home/peter/ply')
import numpy as np
from tensor import Tensor
from runtime import run

# Build and run the Ply program to get initial data + parameters
src = '''
N := 100
X := randn(N, 1)
Y := X * 3.0 + randn(N, 1) * 0.1
W := randn(1, 1) * 0.1
b := zeros(1)
X
'''
rt = run.__wrapped__ if hasattr(run, '__wrapped__') else run
# Can't get intermediates easily, so let's use the Runtime directly
from parser import parse
from runtime import Runtime

program = parse(src)
rt = Runtime()
result = rt.eval(program)
X = rt.env['X']
Y = rt.env['Y']
W = rt.env['W']
b = rt.env['b']

lr = 0.01
for epoch in range(100):
    # Forward: this Ply snippet gets re-evaluated each iteration
    pred = X @ W + b
    diff = pred - Y
    loss = (diff * diff).mean()

    # Gradient
    W.grad = None
    b.grad = None
    loss.backward()

    # Update
    W.data -= lr * W.grad
    b.data -= lr * b.grad

    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d}: loss={loss.data.item():.6f}, W={W.data.item():.4f}, b={b.data.item():.4f}')

print(f'Final: W={W.data.item():.4f} (true=3.0), b={b.data.item():.4f} (true=0.0)')
