"""Loss functions with their own gradients (forward *and* backward).

Both losses return ``(value, gradient)`` so a caller never has to re-derive
the derivative by hand — and so the gradient can be gradient-checked against
a central difference of the scalar value.

cross_entropy (numerically stable)
----------------------------------
With logits ``z`` of shape ``(B, C)`` and integer labels ``y`` of shape
``(B,)``, the mean reduction is::

    ℓ = − (1/B) Σ_b log p_{y_b},
    p = softmax(z)

Evaluating ``log softmax`` directly as ``log(exp(z)/Σ exp(z))`` overflows for
large ``z``. We therefore compute the log-sum-exp trick::

    m_b  = max_c z_{b,c}
    ℓ_b  = − (1/B) Σ_b [ z_{b,y_b} − m_b − log Σ_c exp(z_{b,c} − m_b) ]

and because ``∂ℓ/∂z = (softmax(z) − one_hot(y)) / B``, the gradient is
available from the same stable intermediate::

    dz = (p − one_hot(y)) / B

This is the exact gradient of the *fused* softmax+cross-entropy that
``torch.nn.CrossEntropyLoss`` computes, which is why the classifier heads in
:mod:`dlfs.nn` emit raw logits and never apply a Softmax layer during
training.

mse
---
With ``n = pred.size`` (all elements, mean reduction)::

    ℓ = (1/n) Σ (pred − target)²
    dpred = 2 (pred − target) / n
"""

from __future__ import annotations

import numpy as np

__all__ = ["cross_entropy", "mse", "softmax"]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over the last axis."""
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean softmax cross-entropy over raw logits.

    Parameters
    ----------
    logits:
        Unnormalised scores, shape ``(B, C)``.
    targets:
        Integer class labels of shape ``(B,)`` (or a one-hot array of shape
        ``(B, C)``, which is converted internally).

    Returns
    -------
    (loss, dlogits):
        The scalar mean loss and ``∂loss/∂logits`` of the same shape as
        ``logits``.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"cross_entropy expects 2-D logits (B, C), got {z.shape}")
    y = np.asarray(targets)
    if y.ndim == 2:
        y = y.argmax(axis=-1)
    y = y.astype(np.int64).reshape(-1)
    if y.shape[0] != z.shape[0]:
        raise ValueError(f"got {y.shape[0]} labels for {z.shape[0]} logits")
    if z.shape[0] == 0:
        raise ValueError("cross_entropy received an empty batch")

    # log-sum-exp trick: subtract the row max so exp() can never overflow.
    row_max = z.max(axis=-1, keepdims=True)
    shifted = z - row_max
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp  # log softmax(z)

    batch = z.shape[0]
    picked = log_probs[np.arange(batch), y]
    loss = float(-picked.mean())

    probs = np.exp(log_probs)
    dlogits = probs.copy()
    dlogits[np.arange(batch), y] -= 1.0
    dlogits /= batch
    return loss, dlogits


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error with its gradient.

    ``loss = mean((pred - target)**2)`` over *all* elements, so the gradient
    is ``2 (pred - target) / pred.size``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"mse shape mismatch: {pred.shape} vs {target.shape}")
    if pred.size == 0:
        raise ValueError("mse received empty arrays")
    diff = pred - target
    loss = float(np.mean(diff**2))
    grad = (2.0 / diff.size) * diff
    return loss, grad
