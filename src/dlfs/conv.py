"""Convolution and pooling layers implemented via im2col."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["Conv2D", "MaxPool2D", "col2im", "im2col"]


def _pair(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def im2col(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> np.ndarray:
    """Extract sliding local blocks into columns.

    Input ``x`` is ``(N, C, H, W)``. Returns array of shape
    ``(N, C*kh*kw, out_h*out_w)``.
    """
    x = np.asarray(x, dtype=np.float64)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")

    N, C, H, W = x.shape
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    cols = np.zeros((N, C * kh * kw, out_h * out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            cols[:, :, i * out_w + j] = patch.reshape(N, -1)
    return cols


def col2im(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> np.ndarray:
    """Inverse of :func:`im2col` (accumulates overlapping gradients)."""
    N, C, H, W = x_shape
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    H_p, W_p = H + 2 * ph, W + 2 * pw
    out_h = (H_p - kh) // sh + 1
    out_w = (W_p - kw) // sw + 1

    x_padded = np.zeros((N, C, H_p, W_p), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = cols[:, :, i * out_w + j].reshape(N, C, kh, kw)
            x_padded[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += patch

    if ph or pw:
        return x_padded[:, :, ph : ph + H, pw : pw + W]
    return x_padded


class Conv2D:
    """2-D convolution (NCHW)."""

    name = "conv2d"

    def __init__(
        self,
        *args,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        rng: Optional[np.random.Generator] = None,
        init: str = "he",
        bias: bool = True,
        **kwargs,
    ) -> None:
        # positional parsing
        args = list(args)
        if len(args) == 3:
            in_channels = args[0] if in_channels is None else in_channels
            out_channels = args[1] if out_channels is None else out_channels
            kernel_size = args[2] if kernel_size is None else kernel_size
        elif len(args) == 2:
            if in_channels is None:
                in_channels = None
            out_channels = args[0] if out_channels is None else out_channels
            kernel_size = args[1] if kernel_size is None else kernel_size
        elif len(args) == 1:
            kernel_size = args[0] if kernel_size is None else kernel_size

        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.stride = stride
        self.padding = padding
        self.init = init
        self.bias = bias
        self.kernel_size = kernel_size if kernel_size is not None else 3
        self.out_channels = out_channels
        self.in_channels = in_channels
        self._kh, self._kw = _pair(self.kernel_size)
        self._sh, self._sw = _pair(self.stride)
        self._ph, self._pw = _pair(self.padding)

        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        self._initialized = False
        self._cache = None

        if in_channels is not None and out_channels is not None:
            self._init_weights(int(in_channels))

    def _init_weights(self, in_channels: int) -> None:
        if self._initialized:
            return
        self.in_channels = int(in_channels)
        fan_in = self.in_channels * self._kh * self._kw
        if self.init == "he":
            scale = float(np.sqrt(2.0 / fan_in))
        elif self.init == "glorot":
            scale = float(np.sqrt(2.0 / (fan_in + self.out_channels)))
        else:
            scale = float(np.sqrt(1.0 / fan_in))
        self.W = self.rng.normal(0.0, scale, (self.out_channels, self.in_channels, self._kh, self._kw))
        self.b = np.zeros(self.out_channels, dtype=np.float64) if self.bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if self.b is not None else None
        self._initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects (N,C,H,W), got {x.shape}")
        if not self._initialized:
            self._init_weights(x.shape[1])
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2D expected {self.in_channels} channels, got {C}")

        cols = im2col(x, self.kernel_size, self.stride, self.padding)  # (N, K, L)
        W_col = self.W.reshape(self.out_channels, -1)  # (F, K)
        out = np.einsum("fk,nkl->nfl", W_col, cols)  # (N, F, L)
        if self.b is not None:
            out = out + self.b[None, :, None]
        out_h = (H + 2 * self._ph - self._kh) // self._sh + 1
        out_w = (W + 2 * self._pw - self._kw) // self._sw + 1
        out = out.reshape(N, self.out_channels, out_h, out_w)

        self._cache = (x.shape, cols)
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Conv2D.backward called before forward")
        x_shape, cols = self._cache
        dy = np.asarray(dy, dtype=np.float64)
        N, F, Hout, Wout = dy.shape
        L = Hout * Wout
        dy_reshaped = dy.reshape(N, F, L)

        # dW
        self.dW = np.einsum("nfl,nkl->fk", dy_reshaped, cols).reshape(self.W.shape)
        if self.b is not None:
            self.db = dy_reshaped.sum(axis=(0, 2))

        # dcols
        W_col = self.W.reshape(F, -1)  # (F, K)
        dcols = np.einsum("fk,nfl->nkl", W_col, dy_reshaped)  # (N, K, L)

        dx = col2im(dcols, x_shape, self.kernel_size, self.stride, self.padding)
        return dx

    def params(self) -> dict[str, np.ndarray]:
        if not self._initialized:
            return {}
        out = {"W": self.W}
        if self.b is not None:
            out["b"] = self.b
        return out

    def grads(self) -> dict[str, np.ndarray]:
        if not self._initialized:
            return {}
        out = {"W": self.dW}
        if self.b is not None:
            out["b"] = self.db
        return out

    def __repr__(self) -> str:
        return f"Conv2D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size})"


class MaxPool2D:
    name = "maxpool2d"

    def __init__(
        self,
        *args,
        pool_size: Union[int, Tuple[int, int]] = 2,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        **kwargs,
    ) -> None:
        args = list(args)
        if len(args) >= 1 and kernel_size is None and pool_size == 2 and not kwargs.get("_pool_set", False):
            # ambiguous: first positional could be pool_size
            pass
        if len(args) >= 1:
            pool_size = args[0]
        if kernel_size is not None:
            pool_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        if stride is None:
            stride = pool_size
        self.stride = stride
        self._kh, self._kw = _pair(self.pool_size)
        self._sh, self._sw = _pair(self.stride)
        self._ph, self._pw = _pair(self.padding)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 4:
            raise ValueError(f"MaxPool2D expects (N,C,H,W), got {x.shape}")
        N, C, H, W = x.shape
        ph, pw = self._ph, self._pw
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
        H_p, W_p = x.shape[2], x.shape[3]
        out_h = (H_p - self._kh) // self._sh + 1
        out_w = (W_p - self._kw) // self._sw + 1
        out = np.zeros((N, C, out_h, out_w), dtype=np.float64)
        # store argmax flat indices into original (padded) array
        self._cache = (x.shape, out_h, out_w, H, W, ph, pw)
        self._x_padded = x
        self._out_h = out_h
        self._out_w = out_w
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i * self._sh : i * self._sh + self._kh, j * self._sw : j * self._sw + self._kw]
                maxval = region.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = maxval.squeeze((2, 3))
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dy = np.asarray(dy, dtype=np.float64)
        N, C, Hout, Wout = dy.shape
        x = self._x_padded
        dx_padded = np.zeros_like(x)
        for i in range(Hout):
            for j in range(Wout):
                region = x[:, :, i * self._sh : i * self._sh + self._kh, j * self._sw : j * self._sw + self._kw]
                maxval = region.max(axis=(2, 3), keepdims=True)
                ismax = (region == maxval)
                count = ismax.sum(axis=(2, 3), keepdims=True)
                ismax = ismax / np.where(count == 0, 1, count)
                upstream = dy[:, :, i, j][:, :, None, None]
                dx_padded[:, :, i * self._sh : i * self._sh + self._kh, j * self._sw : j * self._sw + self._kw] += ismax * upstream
        # crop padding
        ph, pw = self._ph, self._pw
        H, W = self._cache[3], self._cache[4]
        if ph or pw:
            return dx_padded[:, :, ph : ph + H, pw : pw + W]
        return dx_padded

    def params(self) -> dict:
        return {}

    def grads(self) -> dict:
        return {}