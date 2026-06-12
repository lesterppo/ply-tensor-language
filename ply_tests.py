"""
ply_tests.py — Comprehensive test suite for Ply language v2.0.
Tests: tensor ops, autodiff, optimizers, parser, runtime, IR compiler.
Run: python3 ply_tests.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tensor import Tensor, param, tensor_relu, tensor_gelu, tensor_sigmoid, tensor_tanh
from tensor import tensor_softmax, tensor_layernorm, tensor_where, tensor_einsum
from optim import SGD, Adam, AdamW
from parser import parse, ParseError
from tokens import tokenize, TokenType
from runtime import Runtime, run as ply_run


def assert_close(a, b, msg='', tol=1e-5):
    a_val = a.data if isinstance(a, Tensor) else a
    b_val = b.data if isinstance(b, Tensor) else b
    if not np.allclose(a_val, b_val, atol=tol):
        raise AssertionError(f"{msg}: {a_val} != {b_val}")


def test_tensor_basics():
    """Tensor creation and basic ops."""
    a = Tensor([1.0, 2.0, 3.0])
    assert a.shape == (3,)
    assert a.dtype_str == 'float32'

    b = a + a
    assert_close(b, Tensor([2.0, 4.0, 6.0]))

    c = a * Tensor([2.0, 2.0, 2.0])
    assert_close(c, Tensor([2.0, 4.0, 6.0]))

    d = a @ Tensor([1.0, 1.0, 1.0]).reshape(3, 1)
    assert_close(d, Tensor([[6.0]]))

    print("  ✓ tensor basics")


def test_autodiff():
    """Automatic differentiation correctness."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 3.0, 4.0])
    c = (a * b).sum()
    c.backward()

    # d/da (a*b).sum() = b
    assert_close(a.grad, Tensor([2.0, 3.0, 4.0]))
    # d/db (a*b).sum() = a
    assert_close(b.grad, Tensor([1.0, 2.0, 3.0]))

    print("  ✓ autodiff (mul+sum)")


def test_activations():
    """Activation function forward/backward."""
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU
    r = tensor_relu(x)
    assert_close(r, Tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    r.sum().backward()
    assert_close(x.grad, Tensor([0.0, 0.0, 0.0, 1.0, 1.0]))

    # Sigmoid
    s = tensor_sigmoid(Tensor([0.0]))
    assert abs(s.data.item() - 0.5) < 1e-5

    # Tanh
    t = tensor_tanh(Tensor([0.0]))
    assert abs(t.data.item()) < 1e-5

    print("  ✓ activations (ReLU, Sigmoid, Tanh)")


def test_softmax():
    """Softmax properties."""
    x = Tensor([1.0, 2.0, 3.0])
    s = tensor_softmax(x, dim=0)
    # Sum to 1
    assert abs(s.data.sum() - 1.0) < 1e-5
    # Monotonic
    assert s.data[0] < s.data[1] < s.data[2]

    print("  ✓ softmax")


def test_layernorm():
    """Layer norm properties."""
    x = Tensor(np.random.randn(4, 8).astype(np.float32))
    ln = tensor_layernorm(x)
    # Mean ~0, std ~1 along last axis
    means = ln.data.mean(axis=-1)
    stds = ln.data.std(axis=-1)
    assert np.allclose(means, 0, atol=1e-5)
    assert np.allclose(stds, 1, atol=1e-4)

    print("  ✓ layernorm")


def test_matmul_grad():
    """Matrix multiply gradient."""
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = (A @ B).sum()
    C.backward()

    # d/dA sum(A@B) = B.T summed appropriately
    expected_A = np.ones_like(A.data) @ B.data.T
    assert_close(A.grad, Tensor(expected_A))

    print("  ✓ matmul gradient")


def test_param():
    """Parameter creation."""
    p = param(np.array([1.0, 2.0, 3.0]), name='test')
    assert p.is_param
    assert p.name == 'test'
    assert p.is_leaf
    assert p._op == 'param'

    print("  ✓ param creation")


def test_optimizer_sgd():
    """SGD optimizer step."""
    w = param(np.array([1.0, 2.0, 3.0]), name='w')
    x = Tensor(np.array([1.0, 1.0, 1.0]))
    y = Tensor(np.array([2.0, 4.0, 6.0]))

    pred = w * x
    diff = pred - y
    loss = (diff * diff).mean()

    opt = SGD([w], lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # w should move towards y/x = [2,4,6]
    assert w.data[0] > 1.0  # moved up
    assert w.data[1] > 2.0
    assert w.data[2] > 3.0

    print("  ✓ SGD optimizer")


def test_optimizer_adam():
    """Adam optimizer step on simple regression."""
    w = param(np.array([0.0]), name='w')
    x = Tensor(np.array([[1.0]]))
    y = Tensor(np.array([[3.0]]))

    opt = Adam([w], lr=0.1)
    for _ in range(100):
        pred = x @ w.reshape(1, 1)
        diff = pred - y
        loss = (diff * diff).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Should approach 3.0
    assert abs(w.data.item() - 3.0) < 0.1
    print("  ✓ Adam optimizer (converged to", round(w.data.item(), 2), ")")


def test_tokenizer():
    """Lexer correctness."""
    tokens = tokenize("A := B @ C + 3.14\n")
    types = [t.type for t in tokens]
    assert types == [TokenType.ID, TokenType.BIND, TokenType.ID, TokenType.AT,
                     TokenType.ID, TokenType.PLUS, TokenType.FLOAT, TokenType.NEWLINE, TokenType.EOF]
    print("  ✓ tokenizer")


def test_parser():
    """Parser correctness."""
    prog = parse("A := randn(3, 4)\nB := A @ A.T\nB")
    assert len(prog.bindings) == 2
    assert prog.bindings[0].name == 'A'
    assert prog.bindings[1].name == 'B'
    assert prog.result is not None
    print("  ✓ parser")


def test_parser_attention():
    """Parser handles attention program."""
    src = open('examples/attention.ply').read()
    prog = parse(src)
    assert len(prog.bindings) > 10
    assert prog.result is not None
    print("  ✓ parser (attention.ply)")


def test_runtime():
    """Runtime executes programs."""
    result = ply_run("X := ones(2, 3)\nsum(X)")
    assert_close(result, Tensor(6.0))
    print("  ✓ runtime")


def test_runtime_ternary():
    """Runtime handles ternary."""
    result = ply_run("a := arange(1, 3)\nb := arange(10, 12)\nm := a > 1.5\nm ? a : b")
    assert_close(result, Tensor([10.0, 2.0]))
    print("  ✓ runtime (ternary)")


def test_ir_compiler():
    """IR compiler produces correct results."""
    from ir import compile_and_run

    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    c = a + b
    d = c * Tensor(np.array([2.0, 2.0, 2.0]))
    e = tensor_relu(d)

    ir_result = compile_and_run(e, apply_fusion=True)
    assert np.allclose(e.data, ir_result)
    print("  ✓ IR compiler (fused)")


def test_einsum():
    """Einsum correctness."""
    A = Tensor(np.random.randn(3, 4).astype(np.float32))
    B = Tensor(np.random.randn(4, 5).astype(np.float32))
    C = tensor_einsum('ij,jk->ik', A, B)
    expected = A.data @ B.data
    assert np.allclose(C.data, expected)
    print("  ✓ einsum")


def test_save_load():
    """Save and load tensors."""
    w = Tensor(np.array([1.0, 2.0, 3.0]), name='weights')
    np.savez('/tmp/ply_test.npz', weights=w.data)
    loaded = np.load('/tmp/ply_test.npz')
    assert np.allclose(loaded['weights'], w.data)
    print("  ✓ save/load")


def main():
    tests = [
        test_tensor_basics,
        test_autodiff,
        test_activations,
        test_softmax,
        test_layernorm,
        test_matmul_grad,
        test_param,
        test_optimizer_sgd,
        test_optimizer_adam,
        test_tokenizer,
        test_parser,
        test_parser_attention,
        test_runtime,
        test_runtime_ternary,
        test_ir_compiler,
        test_einsum,
        test_save_load,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
