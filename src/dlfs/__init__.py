"""dlfs — deep learning built from scratch on top of NumPy.

Why this package exists
-----------------------
I use PyTorch every day in my Honours research (CNN-LSTM models), but ``loss.
backward()`` and ``optimizer.step()`` are black boxes there. This package
implements the forward *and* backward pass of the same building blocks —
dense layers, activations, conv2d, max-pool, an LSTM cell, cross-entropy,
SGD and Adam — in plain NumPy, and proves the backwards passes are correct
with central-difference gradient checks.

Nothing here is meant to be fast, general or production-grade. See the
repository ``README.md`` (especially *Limitations*).
"""

from __future__ import annotations

from dlfs.conv import Conv2D, MaxPool2D, col2im, im2col
from dlfs.gradcheck import (
    DEFAULT_EPS,
    DEFAULT_TOL,
    CheckResult,
    numerical_gradient,
    relative_error,
)
from dlfs.nn import Sequential
from dlfs.rnn import LSTM, LSTMCell
from dlfs.tensors.layers import (
    Dense,
    Flatten,
    LastTimeStep,
    Layer,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)
from dlfs.tensors.losses import cross_entropy, mse, softmax
from dlfs.tensors.optim import SGD, Adam, Optimizer, clip_global_norm

__version__ = "0.1.0"

__all__ = [
    "Adam",
    "CheckResult",
    "Conv2D",
    "DEFAULT_EPS",
    "DEFAULT_TOL",
    "Dense",
    "Flatten",
    "LSTM",
    "LSTMCell",
    "LastTimeStep",
    "Layer",
    "MaxPool2D",
    "Optimizer",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "clip_global_norm",
    "col2im",
    "cross_entropy",
    "im2col",
    "mse",
    "numerical_gradient",
    "relative_error",
    "softmax",
    "__version__",
]
