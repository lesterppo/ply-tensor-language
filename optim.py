"""
Ply Optimizers — SGD, Adam, AdamW.
Operate on Tensor parameters via in-place data updates.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = [p for p in params if p.is_param]
        self.lr = lr
        self._t = 0  # step counter

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self._t += 1
        self._step_impl()

    def _step_impl(self):
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {'lr': self.lr, 'step': self._t}

    def load_state_dict(self, d: Dict[str, Any]):
        self.lr = d['lr']
        self._t = d['step']


class SGD(Optimizer):
    """SGD with optional momentum and weight decay."""
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._v: Dict[int, Any] = {}   # momentum buffers

    def _step_impl(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.copy()

            if self.weight_decay > 0:
                g += self.weight_decay * p.data

            if self.momentum > 0:
                pid = id(p)
                if pid not in self._v:
                    self._v[pid] = np.zeros_like(p.data)
                self._v[pid] = self.momentum * self._v[pid] + g
                g = self._v[pid]

            p.data_sub_(self.lr * g)

    def state_dict(self):
        d = super().state_dict()
        d.update({'momentum': self.momentum, 'weight_decay': self.weight_decay})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.momentum = d.get('momentum', 0.0)
        self.weight_decay = d.get('weight_decay', 0.0)


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba 2015)."""
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: Dict[int, Any] = {}  # first moment
        self._v: Dict[int, Any] = {}  # second moment

    def _step_impl(self):
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            g = p.grad.copy()

            if self.weight_decay > 0:
                g += self.weight_decay * p.data

            if pid not in self._m:
                self._m[pid] = np.zeros_like(p.data)
                self._v[pid] = np.zeros_like(p.data)

            self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * g
            self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * (g ** 2)

            m_hat = self._m[pid] / (1 - self.beta1 ** self._t)
            v_hat = self._v[pid] / (1 - self.beta2 ** self._t)

            p.data_sub_(self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def state_dict(self):
        d = super().state_dict()
        d.update({'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps,
                   'weight_decay': self.weight_decay})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.beta1 = d.get('beta1', 0.9)
        self.beta2 = d.get('beta2', 0.999)
        self.eps = d.get('eps', 1e-8)
        self.weight_decay = d.get('weight_decay', 0.0)


class AdamW(Optimizer):
    """AdamW — Adam with decoupled weight decay (Loshchilov & Hutter 2019)."""
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: Dict[int, Any] = {}
        self._v: Dict[int, Any] = {}

    def _step_impl(self):
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)

            if pid not in self._m:
                self._m[pid] = np.zeros_like(p.data)
                self._v[pid] = np.zeros_like(p.data)

            g = p.grad.copy()

            # Decoupled weight decay: apply directly to params
            if self.weight_decay > 0:
                p.data_mul_(1.0 - self.lr * self.weight_decay)

            self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * g
            self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * (g ** 2)

            m_hat = self._m[pid] / (1 - self.beta1 ** self._t)
            v_hat = self._v[pid] / (1 - self.beta2 ** self._t)

            p.data_sub_(self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def state_dict(self):
        d = super().state_dict()
        d.update({'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps,
                   'weight_decay': self.weight_decay})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.beta1 = d.get('beta1', 0.9)
        self.beta2 = d.get('beta2', 0.999)
        self.eps = d.get('eps', 1e-8)
        self.weight_decay = d.get('weight_decay', 0.01)


_OPTIMIZER_MAP = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
}


def create_optimizer(name: str, params: List[Tensor], **kwargs) -> Optimizer:
    """Create optimizer by name: 'sgd', 'adam', 'adamw'."""
    if name not in _OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer: '{name}'. Available: {list(_OPTIMIZER_MAP)}")
    return _OPTIMIZER_MAP[name](params, **kwargs)
