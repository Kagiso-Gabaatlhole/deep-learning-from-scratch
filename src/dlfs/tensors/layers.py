"""Dense layers and elementwise activations — forward **and** backward.

Layer contract
--------------
Every layer in this module implements the same two-method protocol::

    y    = layer.forward(x)     # caches whatever backward() will need
    dx   = layer.backward(dy)   # dL/dx, and stores dL/dparam in self.grads()

``backward`` must return the gradient with respect to the **input** so that
gradients chain through an arbitrary stack of layers (this is exactly what
``torch.autograd`` does when it walks the graph backwards).

Backprop equations implemented here
-----------------------------------
Dense / linear layer.  With input ``x`` of shape ``(B, ..., F)``, weights
``W`` of shape ``(F, G)`` and bias ``b`` of shape ``(G,)``::

    y = x @ W + b

    dy                     # upstream gradient, shape (B, ..., G)
    dW = sum_{batch axes}( xᵀ · dy )      shape (F, G)
    db = sum_{batch axes}( dy )           shape (G,)
    dx = dy @ Wᵀ                         shape (B, ..., F)

"Batch axes" means *every axis except the last one*, so the identical code
handles an ``(B, F)`` MLP input and a time-distributed ``(B, T, F)`` input.
``dW`` uses ``np.tensordot(..., axes=([0], [0]))`` which sums over axis 0 of
both operands — the batch contraction above.

ReLU    ``y = max(0, x)``                ``dx = dy · 1[x > 0]``
Sigmoid ``y = 1 / (1 + e^{-x})``         ``dx = dy · y · (1 - y)``
Tanh    ``y = tanh(x)``                  ``dx = dy · (1 - y²)``

Softmax (applied row-wise over the last axis)
    ``y_i = e^{x_i} / Σ_j e^{x_j}``

    The Jacobian of one row is ``J = diag(y) − y yᵀ``, so the vector-Jacobian
    product needed for backprop is::

        dx_i = y_i · ( dy_i − Σ_j dy_j y_j )

    which is O(n) instead of materialising the n×n Jacobian.

Numerical stability notes
-------------------------
* ``Sigmoid`` evaluates ``e^{-|x|}`` and branches on the sign of ``x`` so the
  exponential can never overflow.
* ``Softmax`` subtracts the row maximum before exponentiating.
* ``cross_entropy`` in :mod:`dlfs.tensors.losses` is fused with softmax and
  also subtracts the row maximum — that is why :class:`Softmax` is *not*
  wired into the forward pass of the classifier heads in :mod:`dlfs.nn`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Dense",
    "Flatten",
    "LastTimeStep",
    "Layer",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
]

_INIT_MODES = ("he", "glorot", "lecun")


class Layer(ABC):
    """Base class: a differentiable map with cached forward intermediates."""

    name: str = "layer"

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass and cache intermediates for :meth:`backward`."""

    @abstractmethod
    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Return ``dL/dx`` given the upstream gradient ``dy = dL/dy``."""

    def params(self) -> dict[str, np.ndarray]:
        """Named trainable parameters (empty for stateless layers)."""
        return {}

    def grads(self) -> dict[str, np.ndarray]:
        """Gradients matching the keys of :meth:`params` (empty if none)."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


def _init_scale(fan_in: int, mode: str, fan_out: int | None = None) -> float:
    """Return the standard deviation for the requested initialiser.

    * ``he``     — ``sqrt(2 / fan_in)`` (He et al., for ReLU-family nets)
    * ``glorot`` — ``sqrt(2 / (fan_in + fan_out))`` (Glorot & Bengio)
    * ``lecun``  — ``sqrt(1 / fan_in)`` (LeCun, for tanh/sigmoid/linear)
    """
    if mode not in _INIT_MODES:
        raise ValueError(f"init must be one of {_INIT_MODES}, got {mode!r}")
    if mode == "he":
        return float(np.sqrt(2.0 / fan_in))
    if mode == "glorot":
        return float(np.sqrt(2.0 / (fan_in + (fan_out or fan_in))))
    return float(np.sqrt(1.0 / fan_in))


class Dense(Layer):
    """Fully connected layer ``y = x @ W + b``.

    Parameters
    ----------
    in_features:
        Size of the last axis of the input.
    out_features:
        Size of the last axis of the output.
    rng:
        ``numpy.random.Generator`` used to draw ``W`` (keeps every model in
        the repo reproducible from a single seed).
    bias:
        Whether to learn an additive bias ``b``.
    init:
        ``"he"`` (default, for ReLU nets), ``"glorot"`` or ``"lecun"``.
    """

    name = "dense"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        *,
        bias: bool = True,
        init: str = "he",
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init = init
        scale = _init_scale(self.in_features, init, self.out_features)
        self.W: np.ndarray = rng.normal(0.0, scale, (self.in_features, self.out_features))
        self.b: np.ndarray | None = (
            np.zeros(self.out_features, dtype=np.float64) if bias else None
        )
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray | None = np.zeros_like(self.b) if self.b is not None else None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``(B, ..., F) -> (B, ..., G)``."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Dense expected last axis {self.in_features}, got {x.shape[-1]}"
            )
        self._x = x
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Chain rule: returns ``dx`` and fills ``self.dW`` / ``self.db``."""
        if self._x is None:  # pragma: no cover - defensive
            raise RuntimeError("Dense.backward called before forward")
        x2 = np.asarray(self._x, dtype=np.float64).reshape(-1, self.in_features)
        dy2 = np.asarray(dy, dtype=np.float64).reshape(-1, self.out_features)
        self.dW = x2.T @ dy2
        if self.b is not None:
            self.db = dy2.sum(axis=0)
        return np.asarray(dy, dtype=np.float64) @ self.W.T

    def params(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"W": self.W}
        if self.b is not None:
            out["b"] = self.b
        return out

    def grads(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"W": self.dW}
        if self.b is not None:
            out["b"] = self.db  # type: ignore[dict-item]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dense({self.in_features}, {self.out_features}, "
            f"bias={self.b is not None}, init={self.init!r})"
        )


class _Elementwise(Layer):
    """Shared plumbing for activations whose Jacobian is diagonal."""

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, dy: np.ndarray) -> np.ndarray: ...


class ReLU(_Elementwise):
    """``y = max(0, x)``; ``dx = dy · 1[x > 0]`` (subgradient 0 at the kink)."""

    name = "relu"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._y = np.maximum(x, 0.0)
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:  # pragma: no cover - defensive
            raise RuntimeError("ReLU.backward called before forward")
        return dy * (self._y > 0.0)


class Sigmoid(_Elementwise):
    """``y = 1/(1+e^{-x})``; ``dx = dy · y (1 − y)``, overflow-safe."""

    name = "sigmoid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # e^{-|x|} is always in (0, 1], so this never overflows.
        exp_neg_abs = np.exp(-np.abs(x))
        self._y = np.where(x >= 0.0, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:  # pragma: no cover - defensive
            raise RuntimeError("Sigmoid.backward called before forward")
        return dy * self._y * (1.0 - self._y)


class Tanh(_Elementwise):
    """``y = tanh(x)``; ``dx = dy · (1 − y²)``."""

    name = "tanh"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.tanh(np.asarray(x, dtype=np.float64))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:  # pragma: no cover - defensive
            raise RuntimeError("Tanh.backward called before forward")
        return dy * (1.0 - self._y**2)


class Softmax(_Elementwise):
    """Row-wise softmax over the last axis with a stable forward pass.

    Backward applies the full row Jacobian ``diag(y) − y yᵀ`` as a
    vector-Jacobian product (see the module docstring). In practice the
    fused ``cross_entropy(logits, labels)`` is used for training because it
    never materialises probabilities; this layer exists so the Jacobian
    itself can be gradient-checked in isolation.
    """

    name = "softmax"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        shifted = x - x.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        self._y = exp / exp.sum(axis=-1, keepdims=True)
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:  # pragma: no cover - defensive
            raise RuntimeError("Softmax.backward called before forward")
        y = self._y
        row = (dy * y).sum(axis=-1, keepdims=True)  # Σ_j dy_j y_j
        return y * (dy - row)


class Flatten(Layer):
    """``(B, d1, d2, ...) -> (B, d1*d2*...)``; restores the shape on backward."""

    name = "flatten"

    def __init__(self) -> None:
        self._shape: tuple[int, ...] = ()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = tuple(np.asarray(x).shape)
        return np.asarray(x, dtype=np.float64).reshape(self._shape[0], -1)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return np.asarray(dy, dtype=np.float64).reshape(self._shape)


class LastTimeStep(Layer):
    """Select ``x[:, -1, :]`` from a ``(B, T, H)`` sequence tensor.

    Backward scatters the upstream gradient into the last time step and
    zeros everywhere else — the discrete analogue of how a gather op
    back-propagates.
    """

    name = "last_timestep"

    def __init__(self) -> None:
        self._shape: tuple[int, ...] = ()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"LastTimeStep expects (B, T, H), got shape {x.shape}")
        self._shape = x.shape
        return x[:, -1, :]

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = np.zeros(self._shape, dtype=np.float64)
        dx[:, -1, :] = dy
        return dx
