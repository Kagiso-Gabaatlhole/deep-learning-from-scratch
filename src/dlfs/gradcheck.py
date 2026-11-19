"""Gradient-checking utilities.

Central (two-sided) differences are used to approximate the derivative of a
scalar function; the analytic gradient produced by a layer is then compared
with the numerical one using the standard relative-error metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

DEFAULT_EPS: float = 1e-5
DEFAULT_TOL: float = 1e-5


@dataclass
class CheckResult:
    """Outcome of a gradient check."""

    layer: str
    max_rel_error: float
    passed: bool
    details: dict[str, float] = field(default_factory=dict)


def relative_error(a: Any, b: Any) -> float:
    """Standard relative error between two arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a) + np.linalg.norm(b))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Central-difference gradient of scalar ``f`` w.r.t. array ``x``."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x, dtype=np.float64)
    flat = x.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        fp = float(f(x))
        flat[i] = orig - eps
        fm = float(f(x))
        flat[i] = orig
        grad.reshape(-1)[i] = (fp - fm) / (2.0 * eps)
    return grad


def _layer_gradient_check(layer, x, eps=DEFAULT_EPS, tol=DEFAULT_TOL, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64).copy()
    y = layer.forward(x)
    dy = rng.standard_normal(np.asarray(y).shape)
    dx_analytic = layer.backward(dy)

    details: dict[str, float] = {}

    def f_x(x_):
        yy = layer.forward(np.asarray(x_, dtype=np.float64))
        return float(np.sum(np.asarray(yy, dtype=np.float64) * dy))

    dx_num = numerical_gradient(f_x, x, eps)
    rel_dx = relative_error(dx_analytic, dx_num)
    details["dx"] = rel_dx
    max_rel = rel_dx

    for name, param in layer.params().items():
        grad = layer.grads()[name]

        def f_param(p_):
            old = np.array(param, dtype=np.float64, copy=True)
            param[...] = np.asarray(p_, dtype=np.float64)
            yy = layer.forward(x)
            val = float(np.sum(np.asarray(yy, dtype=np.float64) * dy))
            param[...] = old
            return val

        num = numerical_gradient(f_param, param, eps)
        rel = relative_error(grad, num)
        details[name] = rel
        if rel > max_rel:
            max_rel = rel

    passed = bool(max_rel < tol)
    return CheckResult(layer=type(layer).__name__, max_rel_error=float(max_rel), passed=passed, details=details)


def gradient_check(layer, x=None, eps=DEFAULT_EPS, tol=DEFAULT_TOL, seed=0, **kwargs):
    if x is None:
        x = kwargs.get("X", kwargs.get("input", None))
    if x is None:
        raise ValueError("gradient_check requires an input array x")
    return _layer_gradient_check(layer, x, eps=eps, tol=tol, seed=seed)


check = gradient_check
run_gradient_check = gradient_check


def check_layer(name: str, seed: int = 0) -> CheckResult:
    from .tensors.layers import Dense, ReLU, Sigmoid, Softmax, Tanh
    from .conv import Conv2D, MaxPool2D
    from .rnn import LSTM

    rng = np.random.default_rng(seed)
    name = name.lower()
    if name == "dense":
        layer = Dense(4, 5, rng=rng)
        x = rng.standard_normal((3, 4))
        return gradient_check(layer, x)
    if name == "relu":
        layer = ReLU()
        x = rng.standard_normal((3, 4))
        return gradient_check(layer, x)
    if name == "sigmoid":
        layer = Sigmoid()
        x = rng.standard_normal((3, 4))
        return gradient_check(layer, x)
    if name == "tanh":
        layer = Tanh()
        x = rng.standard_normal((3, 4))
        return gradient_check(layer, x)
    if name == "softmax":
        layer = Softmax()
        x = rng.standard_normal((3, 4))
        return gradient_check(layer, x)
    if name == "conv":
        layer = Conv2D(2, 3, 3, rng=rng)
        x = rng.standard_normal((2, 2, 6, 6))
        return gradient_check(layer, x)
    if name in ("maxpool", "max_pool", "maxpool2d"):
        layer = MaxPool2D(2)
        x = rng.standard_normal((2, 2, 6, 6))
        return gradient_check(layer, x)
    if name == "lstm":
        layer = LSTM(4, 5, rng=rng, return_sequences=False)
        x = rng.standard_normal((2, 6, 4))
        return gradient_check(layer, x)
    raise ValueError(f"unknown layer {name!r}")