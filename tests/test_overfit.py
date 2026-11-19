import numpy as np

from dlfs.nn import Sequential
from dlfs.tensors.layers import Dense, ReLU
from dlfs.conv import Conv2D, MaxPool2D
from dlfs.tensors.layers import Flatten
from dlfs.rnn import LSTM


def test_mlp_overfit_small():
    rng = np.random.default_rng(0)
    # simple linearly separable-ish small set
    X = rng.standard_normal((32, 4))
    y = (X[:, 0] > 0).astype(int)
    model = Sequential(
        [Dense(4, 16, rng=np.random.default_rng(1)), ReLU(), Dense(16, 2, rng=np.random.default_rng(1))],
        seed=1,
    )
    hist = model.fit(X, y, epochs=200, batch_size=32, lr=1e-2, verbose=False)
    assert hist["loss"][-1] < hist["loss"][0]
    pred = model.predict(X)
    acc = float(np.mean(pred == y))
    assert acc > 0.9


def test_cnn_overfit_small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 1, 8, 8))
    y = (rng.random(32) > 0.5).astype(int)
    model = Sequential(
        [Conv2D(1, 4, 3, rng=np.random.default_rng(1), padding=1), _relu4d(), Flatten(),
         Dense(4 * 8 * 8, 2, rng=np.random.default_rng(1))],
        seed=1,
    )
    hist = model.fit(X, y, epochs=50, batch_size=32, lr=1e-3, verbose=False)
    assert hist["loss"][-1] < hist["loss"][0]


def _relu4d():
    from dlfs.tensors.layers import ReLU

    class _R:
        name = "relu"

        def __init__(self):
            self._l = ReLU()

        def forward(self, x):
            return self._l.forward(x)

        def backward(self, dy):
            return self._l.backward(dy)

        def params(self):
            return {}

        def grads(self):
            return {}

    return _R()


def test_lstm_overfit_small():
    rng = np.random.default_rng(0)
    # constant-ish: label = first bit
    X = rng.integers(0, 2, size=(32, 6, 1)).astype(float)
    y = X[:, 0, 0].astype(int)
    model = Sequential(
        [LSTM(1, 8, rng=np.random.default_rng(1), return_sequences=False),
         Dense(8, 2, rng=np.random.default_rng(1))],
        seed=1,
    )
    hist = model.fit(X, y, epochs=100, batch_size=32, lr=1e-2, verbose=False)
    assert hist["loss"][-1] < hist["loss"][0]
    pred = model.predict(X)
    acc = float(np.mean(pred == y))
    assert acc > 0.9