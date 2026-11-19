"""Gradient-descent optimisers: SGD (momentum) and Adam (with bias correction).

Both optimisers consume two flat dictionaries produced by
``Sequential._param_groups`` — ``{qualified_name: parameter}`` and
``{qualified_name: gradient}`` — and update the parameter arrays **in place**
(the same trick ``torch.optim`` uses, so no re-binding of model references is
needed).

Update rules
------------
SGD with optional momentum ``β`` and L2 weight decay ``λ``::

    v ← β·v + (g + λ·p)          # β = 0 reduces to plain SGD
    p ← p − lr·v

Adam
----
::

    m ← β₁·m + (1 − β₁)·g                    # first-moment estimate
    v ← β₂·v + (1 − β₂)·g²                   # second-moment estimate
    m̂ ← m / (1 − β₁ᵗ)                        # bias correction
    v̂ ← v / (1 − β₂ᵗ)
    p ← p − lr · m̂ / (√v̂ + ε)

Why the bias correction matters: at ``t = 1`` we have ``m = (1 − β₁)g`` and
``v = (1 − β₂)g²``, so *without* dividing by ``(1 − β₁ᵗ)`` and ``(1 − β₂ᵗ)``
both moments start near zero and the first updates are far too small (and
badly scaled, since ``m/v`` is not a unit-free ratio). With the correction,
``t = 1`` collapses to ``m̂ = g``, ``v̂ = g²`` and the step is exactly
``−lr·g / (|g| + ε) ≈ −lr·sign(g)`` — a property the test suite asserts.

Gradient clipping
-----------------
:func:`clip_global_norm` rescales every gradient array by a single factor
``max_norm / ‖g‖₂`` when the global L2 norm of all parameters exceeds
``max_norm``. Clipping the *global* norm (rather than each tensor
independently) preserves the relative direction of the update — this is what
``torch.nn.utils.clip_grad_norm_`` does, and it is standard practice for
stabilising LSTM BPTT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["SGD", "Adam", "Optimizer", "clip_global_norm"]


def clip_global_norm(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    max_norm: float,
) -> float:
    """Scale all gradients so their combined L2 norm is at most ``max_norm``.

    Returns the **pre-clipping** global norm so callers can log how often
    clipping actually fired.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    squares = 0.0
    for key, grad in grads.items():
        if grad is None:  # pragma: no cover - defensive
            continue
        if key not in params:  # pragma: no cover - defensive
            raise KeyError(f"gradient {key!r} has no matching parameter")
        squares += float(np.sum(np.asarray(grad, dtype=np.float64) ** 2))
    norm = float(np.sqrt(squares))
    if norm > max_norm and norm > 0.0:
        scale = max_norm / (norm + 1e-12)
        for key in grads:
            if grads[key] is not None:
                grads[key] = grads[key] * scale
    return norm


class Optimizer(ABC):
    """Base class for in-place parameter updaters."""

    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)
        self.steps = 0

    @abstractmethod
    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        """Apply one update to every entry of ``params`` in place."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(lr={self.lr})"


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocity: dict[str, np.ndarray] = {}

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        self.steps += 1
        for key, param in params.items():
            grad = np.asarray(grads[key], dtype=np.float64)
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            if self.momentum:
                vel = self._velocity.get(key, np.zeros_like(param))
                vel = self.momentum * vel + grad
                self._velocity[key] = vel
                grad = vel
            param -= self.lr * grad


class Adam(Optimizer):
    """Adam with the standard ``β₁``/``β₂`` bias correction (Kingma & Ba, 2015)."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._state: dict[str, dict[str, np.ndarray]] = {}

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        self.steps += 1
        t = self.steps
        # 1 - βᵗ computed as 1 - exp(t·ln β) so large t cannot lose precision
        # to catastrophic cancellation and β₁ᵗ simply underflowing to 0 is the
        # correct limit (bias correction → 1).
        corr1 = 1.0 - np.exp(t * np.log(self.beta1)) if self.beta1 > 0.0 else 1.0
        corr2 = 1.0 - np.exp(t * np.log(self.beta2)) if self.beta2 > 0.0 else 1.0

        for key, param in params.items():
            grad = np.asarray(grads[key], dtype=np.float64)
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            state = self._state.get(key)
            if state is None:
                state = {"m": np.zeros_like(param), "v": np.zeros_like(param)}
                self._state[key] = state
            m = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
            v = self.beta2 * state["v"] + (1.0 - self.beta2) * grad * grad
            state["m"], state["v"] = m, v
            m_hat = m / corr1
            v_hat = v / corr2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
