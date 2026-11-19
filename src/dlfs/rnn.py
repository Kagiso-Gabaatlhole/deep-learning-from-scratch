"""LSTM cell and sequence layer with BPTT."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["LSTM", "LSTMCell"]


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _lstm_step_forward(W, b, x_t, h_prev, c_prev):
    x_t = np.asarray(x_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    c_prev = np.asarray(c_prev, dtype=np.float64)
    combined = np.concatenate([x_t, h_prev], axis=-1)
    z = combined @ W + b
    H = W.shape[1] // 4
    i = _sigmoid(z[:, :H])
    f = _sigmoid(z[:, H : 2 * H])
    g = np.tanh(z[:, 2 * H : 3 * H])
    o = _sigmoid(z[:, 3 * H :])
    c = f * c_prev + i * g
    tanh_c = np.tanh(c)
    h = o * tanh_c
    cache = {
        "combined": combined,
        "i": i,
        "f": f,
        "g": g,
        "o": o,
        "c_prev": c_prev,
        "c": c,
        "tanh_c": tanh_c,
    }
    return h, c, cache


def _lstm_step_backward(W, cache, dh, dc):
    dh = np.asarray(dh, dtype=np.float64)
    dc = np.asarray(dc, dtype=np.float64)
    H = W.shape[1] // 4
    i = cache["i"]
    f = cache["f"]
    g = cache["g"]
    o = cache["o"]
    c_prev = cache["c_prev"]
    c = cache["c"]
    tanh_c = cache["tanh_c"]
    combined = cache["combined"]

    # dc total includes contribution from h
    dc_total = dc + dh * o * (1.0 - tanh_c**2)

    # gate gradients (pre-activation)
    di = dc_total * g
    df = dc_total * c_prev
    dg = dc_total * i
    do = dh * tanh_c

    # activation derivatives
    di = di * i * (1.0 - i)
    df = df * f * (1.0 - f)
    dg = dg * (1.0 - g**2)
    do = do * o * (1.0 - o)

    dz = np.concatenate([di, df, dg, do], axis=-1)  # (B, 4H)

    dcombined = dz @ W.T  # (B, I+H)
    # input_size = W.shape[0]-H? Actually combined = x + h_prev; W rows = I+H
    I = W.shape[0] - (W.shape[1] // 4)  # wrong: W rows = I+H, cols=4H. Need input size separately.
    # We'll infer input size from combined shape and h_prev size = hidden size = H
    Hsize = W.shape[1] // 4
    input_size = combined.shape[1] - Hsize
    dx = dcombined[:, :input_size]
    dh_prev = dcombined[:, input_size:]

    dc_prev = dc_total * f

    dW = combined.T @ dz
    db = dz.sum(axis=0)
    return dx, dh_prev, dc_prev, dW, db


class LSTMCell:
    name = "lstm_cell"

    def __init__(
        self,
        *args,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        init: str = "glorot",
        **kwargs,
    ) -> None:
        args = list(args)
        if len(args) >= 2:
            input_size = args[0] if input_size is None else input_size
            hidden_size = args[1] if hidden_size is None else hidden_size
        elif len(args) == 1:
            hidden_size = args[0] if hidden_size is None else hidden_size

        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.init = init
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._initialized = False
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        self._cache = None
        if input_size is not None and hidden_size is not None:
            self._init(int(input_size), int(hidden_size))

    def _init(self, input_size, hidden_size):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        fan = self.input_size + self.hidden_size
        if self.init == "he":
            scale = float(np.sqrt(2.0 / fan))
        elif self.init == "glorot":
            scale = float(np.sqrt(1.0 / fan))
        else:
            scale = float(np.sqrt(1.0 / fan))
        self.W = self.rng.normal(0.0, scale, (fan, 4 * self.hidden_size))
        self.b = np.zeros(4 * self.hidden_size, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._initialized = True

    def _ensure(self, x):
        if not self._initialized:
            self._init(x.shape[-1], self.hidden_size if self.hidden_size is not None else x.shape[-1])

    def forward(self, x, h_prev=None, c_prev=None):
        x = np.asarray(x, dtype=np.float64)
        if not self._initialized:
            self._ensure(x)
        H = self.hidden_size
        B = x.shape[0]
        if h_prev is None:
            h_prev = np.zeros((B, H), dtype=np.float64)
        if c_prev is None:
            c_prev = np.zeros((B, H), dtype=np.float64)
        h, c, cache = _lstm_step_forward(self.W, self.b, x, h_prev, c_prev)
        self._cache = cache
        return h, c

    def backward(self, dh, dc=None):
        if self._cache is None:
            raise RuntimeError("LSTMCell.backward before forward")
        if dc is None:
            dc = np.zeros_like(dh)
        dx, dh_prev, dc_prev, dW, db = _lstm_step_backward(self.W, self._cache, dh, dc)
        self.dW = dW
        self.db = db
        return dx, dh_prev, dc_prev

    def params(self):
        if not self._initialized:
            return {}
        return {"W": self.W, "b": self.b}

    def grads(self):
        if not self._initialized:
            return {}
        return {"W": self.dW, "b": self.db}


class LSTM:
    name = "lstm"

    def __init__(
        self,
        *args,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        return_sequences: bool = False,
        init: str = "glorot",
        **kwargs,
    ) -> None:
        args = list(args)
        if len(args) >= 2:
            input_size = args[0] if input_size is None else input_size
            hidden_size = args[1] if hidden_size is None else hidden_size
        elif len(args) == 1:
            if hidden_size is None:
                hidden_size = args[0]
            # input_size stays None -> infer

        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.init = init
        self._cell: Optional[LSTMCell] = None
        self._caches = None
        self._x_shape = None
        self._hs = None
        if input_size is not None and hidden_size is not None:
            self._cell = LSTMCell(input_size, hidden_size, rng=rng, init=init)

    def _ensure_cell(self, input_size):
        if self._cell is None:
            if self.hidden_size is None:
                self.hidden_size = self.input_size if self.input_size is not None else input_size
            if self.input_size is None:
                self.input_size = input_size
            self._cell = LSTMCell(self.input_size, self.hidden_size, rng=self.rng, init=self.init)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"LSTM expects (B,T,F), got {x.shape}")
        B, T, F = x.shape
        self._ensure_cell(F)
        self._x_shape = x.shape
        cell = self._cell
        H = cell.hidden_size
        h = np.zeros((B, H), dtype=np.float64)
        c = np.zeros((B, H), dtype=np.float64)
        caches = []
        hs = []
        for t in range(T):
            h_prev = h.copy()
            c_prev = c.copy()
            h, c, cache = _lstm_step_forward(cell.W, cell.b, x[:, t, :], h_prev, c_prev)
            caches.append(cache)
            hs.append(h.copy())
        self._caches = caches
        self._hs = hs
        if self.return_sequences:
            return np.stack(hs, axis=1)
        return hs[-1]

    def backward(self, dy):
        if self._caches is None:
            raise RuntimeError("LSTM.backward before forward")
        x_shape = self._x_shape
        B, T, F = x_shape
        H = self._cell.hidden_size
        dy = np.asarray(dy, dtype=np.float64)
        if self.return_sequences:
            dy_seq = dy
        else:
            dy_seq = np.zeros((B, T, H), dtype=np.float64)
            dy_seq[:, -1, :] = dy

        dx = np.zeros((B, T, F), dtype=np.float64)
        dh_next = np.zeros((B, H), dtype=np.float64)
        dc_next = np.zeros((B, H), dtype=np.float64)
        dW = np.zeros_like(self._cell.W)
        db = np.zeros_like(self._cell.b)
        for t in range(T - 1, -1, -1):
            cache = self._caches[t]
            dh = dy_seq[:, t, :] + dh_next
            dc = dc_next
            step_dx, dh_prev, dc_prev, step_dW, step_db = _lstm_step_backward(self._cell.W, cache, dh, dc)
            dx[:, t, :] = step_dx
            dh_next = dh_prev
            dc_next = dc_prev
            dW += step_dW
            db += step_db
        self._cell.dW = dW
        self._cell.db = db
        return dx

    def params(self):
        if self._cell is None:
            return {}
        return self._cell.params()

    def grads(self):
        if self._cell is None:
            return {}
        return self._cell.grads()

    def __repr__(self):
        return f"LSTM({self.input_size}, {self.hidden_size}, return_sequences={self.return_sequences})"