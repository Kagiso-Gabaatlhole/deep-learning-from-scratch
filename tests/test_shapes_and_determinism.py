import numpy as np

from dlfs.nn import Sequential
from dlfs.tensors.layers import Dense, ReLU, Flatten, LastTimeStep
from dlfs.conv import Conv2D, MaxPool2D
from dlfs.rnn import LSTM, LSTMCell
from dlfs.tensors.losses import cross_entropy, mse
from dlfs.tensors.optim import SGD, Adam, clip_global_norm


def test_dense_shape():
    rng = np.random.default_rng(0)
    layer = Dense(4, 7, rng=rng)
    x = np.zeros((3, 4))
    y = layer.forward(x)
    assert y.shape == (3, 7)
    dx = layer.backward(np.ones((3, 7)))
    assert dx.shape == (3, 4)


def test_dense_time_distributed_shape():
    rng = np.random.default_rng(0)
    layer = Dense(4, 7, rng=rng)
    x = np.zeros((2, 5, 4))
    y = layer.forward(x)
    assert y.shape == (2, 5, 7)
    dx = layer.backward(np.ones((2, 5, 7)))
    assert dx.shape == (2, 5, 4)


def test_conv_shape():
    rng = np.random.default_rng(0)
    layer = Conv2D(2, 4, 3, rng=rng, padding=1)
    x = np.zeros((2, 2, 8, 8))
    y = layer.forward(x)
    assert y.shape == (2, 4, 8, 8)


def test_maxpool_shape():
    layer = MaxPool2D(2)
    x = np.zeros((2, 2, 8, 8))
    y = layer.forward(x)
    assert y.shape == (2, 2, 4, 4)


def test_lstm_shapes():
    rng = np.random.default_rng(0)
    layer = LSTM(3, 6, rng=rng, return_sequences=True)
    x = np.zeros((2, 5, 3))
    y = layer.forward(x)
    assert y.shape == (2, 5, 6)
    dx = layer.backward(np.ones((2, 5, 6)))
    assert dx.shape == (2, 5, 3)

    layer2 = LSTM(3, 6, rng=rng, return_sequences=False)
    x2 = np.zeros((2, 5, 3))
    y2 = layer2.forward(x2)
    assert y2.shape == (2, 6)


def test_last_timestep_shape():
    layer = LastTimeStep()
    x = np.zeros((2, 5, 6))
    y = layer.forward(x)
    assert y.shape == (2, 6)
    dx = layer.backward(np.ones((2, 6)))
    assert dx.shape == (2, 5, 6)


def test_determinism_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = Dense(4, 5, rng=rng1)
    b = Dense(4, 5, rng=rng2)
    x = np.ones((2, 4))
    np.testing.assert_allclose(a.forward(x), b.forward(x))
    # model determinism
    def make():
        r = np.random.default_rng(7)
        m = Sequential([Dense(4, 8, rng=r), ReLU(), Dense(8, 3, rng=r)], seed=123)
        return m
    m1 = make()
    m2 = make()
    xx = np.ones((4, 4))
    np.testing.assert_allclose(m1.forward(xx), m2.forward(xx))


def test_cross_entropy_mse():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 3))
    labels = np.array([0, 1, 2, 1])
    loss, g = cross_entropy(logits, labels)
    assert loss > 0
    assert g.shape == logits.shape
    p, gg = mse(logits, logits.copy())
    assert p == 0.0
    assert np.allclose(gg, 0.0)


def test_clip_global_norm():
    params = {"w": np.ones((2, 2))}
    grads = {"w": np.ones((2, 2)) * 10.0}
    norm = clip_global_norm(params, grads, 1.0)
    assert norm > 1.0
    new_norm = float(np.linalg.norm(grads["w"]))
    assert new_norm <= 1.0 + 1e-9


def test_adam_vs_sgd_smoke():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 4))
    y = (X[:, 0] > 0).astype(int)
    m1 = Sequential([Dense(4, 2, rng=np.random.default_rng(1))], seed=1)
    m2 = Sequential([Dense(4, 2, rng=np.random.default_rng(1))], seed=1)
    # same init
    np.testing.assert_allclose(m1.layers[0].W, m2.layers[0].W)
    h1 = m1.fit(X, y, epochs=3, batch_size=16, optimizer=Adam(lr=1e-2), verbose=False)
    h2 = m2.fit(X, y, epochs=3, batch_size=16, optimizer=SGD(lr=1e-2), verbose=False)
    assert len(h1["loss"]) == 3
    assert len(h2["loss"]) == 3