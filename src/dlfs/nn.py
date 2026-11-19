"""Minimal sequential model with fit/predict."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .tensors.layers import Layer
from .tensors.losses import cross_entropy, mse
from .tensors.optim import Optimizer, Adam


class Sequential:
    def __init__(self, layers, seed: int = 0, rng: Optional[np.random.Generator] = None):
        self.layers = list(layers)
        self.seed = int(seed)
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dy):
        dy = np.asarray(dy, dtype=np.float64)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def params(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params().items():
                out[f"{i}.{getattr(layer, 'name', type(layer).__name__)}.{k}"] = v
        return out

    def parameters(self) -> dict[str, np.ndarray]:
        return self.params()

    def grads(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads().items():
                out[f"{i}.{getattr(layer, 'name', type(layer).__name__)}.{k}"] = v
        return out

    def predict(self, x):
        out = self.forward(x)
        out = np.asarray(out)
        if out.ndim >= 2 and out.shape[-1] > 1:
            return np.argmax(out, axis=-1)
        return out

    def fit(
        self,
        X,
        y,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        optimizer: Optional[Optimizer] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        verbose: bool = True,
        loss: str = "cross_entropy",
    ) -> dict[str, list]:
        X = np.asarray(X, dtype=np.float64).copy()
        y = np.asarray(y)
        if optimizer is None:
            optimizer = Adam(lr=lr)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n = X.shape[0]
        history: dict[str, list] = {"loss": []}
        for epoch in range(int(epochs)):
            if shuffle:
                perm = rng.permutation(n)
                X = X[perm]
                y = y[perm]
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n, int(batch_size)):
                xb = X[start : start + int(batch_size)]
                yb = y[start : start + int(batch_size)]
                logits = self.forward(xb)
                if loss == "cross_entropy":
                    val, dlogits = cross_entropy(logits, yb)
                elif loss == "mse":
                    # assume yb integer labels -> one-hot for mse
                    if yb.ndim == 1:
                        ncls = logits.shape[1]
                        onehot = np.zeros((yb.shape[0], ncls), dtype=np.float64)
                        onehot[np.arange(yb.shape[0]), yb.astype(int)] = 1.0
                        target = onehot
                    else:
                        target = np.asarray(yb, dtype=np.float64)
                    val, dlogits = mse(logits, target)
                else:
                    raise ValueError(f"unknown loss {loss!r}")
                self.backward(dlogits)
                params, grads = self.params(), self.grads()
                optimizer.step(params, grads)
                epoch_loss += val * xb.shape[0]
                nb += 1
            epoch_loss /= n
            history["loss"].append(float(epoch_loss))
            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
        return history