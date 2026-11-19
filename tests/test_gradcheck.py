import numpy as np

from dlfs.gradcheck import (
    DEFAULT_TOL,
    CheckResult,
    gradient_check,
    numerical_gradient,
    relative_error,
)
from dlfs.tensors.layers import Dense, ReLU, Sigmoid, Tanh, Softmax
from dlfs.conv import Conv2D, MaxPool2D
from dlfs.rnn import LSTM


def test_relative_error_basic():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    assert relative_error(a, b) == 0.0
    assert relative_error(a, a * 1.01) > 0


def test_numerical_gradient():
    def f(x):
        return float(np.sum(x ** 2))
    x = np.array([1.0, 2.0, 3.0])
    g = numerical_gradient(f, x)
    np.testing.assert_allclose(g, 2 * x, atol=1e-8)


def test_dense_gradcheck():
    rng = np.random.default_rng(0)
    layer = Dense(4, 5, rng=rng)
    x = rng.standard_normal((3, 4))
    res = gradient_check(layer, x)
    assert isinstance(res, CheckResult)
    assert res.passed
    assert res.max_rel_error < 1e-5


def test_relu_gradcheck():
    rng = np.random.default_rng(1)
    layer = ReLU()
    x = rng.standard_normal((3, 4))
    res = gradient_check(layer, x)
    assert res.passed


def test_sigmoid_gradcheck():
    rng = np.random.default_rng(2)
    layer = Sigmoid()
    x = rng.standard_normal((3, 4))
    res = gradient_check(layer, x)
    assert res.passed


def test_tanh_gradcheck():
    rng = np.random.default_rng(3)
    layer = Tanh()
    x = rng.standard_normal((3, 4))
    res = gradient_check(layer, x)
    assert res.passed


def test_softmax_gradcheck():
    rng = np.random.default_rng(4)
    layer = Softmax()
    x = rng.standard_normal((3, 4))
    res = gradient_check(layer, x)
    assert res.passed


def test_conv_gradcheck():
    rng = np.random.default_rng(5)
    layer = Conv2D(2, 3, 3, rng=rng)
    x = rng.standard_normal((2, 2, 6, 6))
    res = gradient_check(layer, x)
    assert res.passed
    assert res.max_rel_error < 1e-5


def test_maxpool_gradcheck():
    rng = np.random.default_rng(6)
    layer = MaxPool2D(2)
    x = rng.standard_normal((2, 2, 6, 6))
    res = gradient_check(layer, x)
    assert res.passed


def test_lstm_gradcheck():
    rng = np.random.default_rng(7)
    layer = LSTM(4, 5, rng=rng, return_sequences=False)
    x = rng.standard_normal((2, 6, 4))
    res = gradient_check(layer, x)
    assert res.passed
    assert res.max_rel_error < 1e-5