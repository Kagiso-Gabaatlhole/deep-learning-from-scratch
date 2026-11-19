# deep-learning-from-scratch

A teaching-scale deep learning library built entirely on NumPy. Every building
block implements **both** the forward and the backward pass in plain Python, and
the backward passes are verified with central-difference gradient checks.

This is not a competitor to PyTorch. It is a learning artifact: the same
operations I call through `loss.backward()` and `optimizer.step()` in my
research, opened up and re-derived by hand.

## Why

Frameworks hide the calculus. This repo makes it visible: Dense, ReLU/sigmoid/
tanh/softmax, Conv2D via im2col, MaxPool2D, an LSTM cell with BPTT, fused
softmax cross-entropy, MSE, SGD and Adam (with bias correction). Each one
exposes `forward()` and `backward()` and ships a gradient check.

## Architecture

| Module | Contents |
|---|---|
| `src/dlfs/tensors/layers.py` | `Layer`, `Dense`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`, `LastTimeStep` |
| `src/dlfs/tensors/losses.py` | `cross_entropy` (stable, fused with softmax), `mse`, `softmax` |
| `src/dlfs/tensors/optim.py` | `Optimizer`, `SGD` (momentum + weight decay), `Adam` (bias correction), `clip_global_norm` |
| `src/dlfs/conv.py` | `Conv2D` (im2col), `MaxPool2D`, `im2col`, `col2im` |
| `src/dlfs/rnn.py` | `LSTMCell`, `LSTM` (BPTT over short sequences) |
| `src/dlfs/nn.py` | `Sequential` model: `fit()`, `predict()`, minibatching, seeded shuffling |
| `src/dlfs/gradcheck.py` | `numerical_gradient`, `relative_error`, `gradient_check`, `CheckResult` |
| `src/dlfs/cli.py` | argparse CLI (`train`, `check`, `list`) |

## Gradient-check philosophy

For a layer, pick a random input `x` and upstream gradient `dy`. Compute the
analytic `dx = layer.backward(dy)` and parameter gradients. Then approximate
`dL/dx` (and `dL/dW`, `dL/db`) with central differences on the scalar
`L = sum(layer.forward(x) * dy)`. Report the relative error

```
||analytic - numeric|| / (||analytic|| + ||numeric||)
```

A correct backward pass gives ~1e-11 on float64. The suite asserts `< 1e-5`.

Observed (from `python -m dlfs check`):

| Layer | max rel error | passed |
|---|---|---|
| Dense | 1.9e-11 | yes |
| Conv2D | 5.4e-11 | yes |
| LSTM | 4.2e-11 | yes |
| MaxPool2D | 2.1e-11 | yes |

## Math notes

**Dense.** `y = x @ W + b`. With batch axes flattened to `N`:
`dW = X^T dY`, `db = sum_b dY_b`, `dx = dY @ W^T`.

**ReLU / sigmoid / tanh.** Elementwise; gradients are `dy * indicator(x>0)`,
`dy * y(1-y)`, `dy * (1-y^2)`.

**Softmax.** Row-wise Jacobian `diag(y) - y y^T`; applied as a
vector-Jacobian product `y * (dy - sum(dy*y))`, O(n) per row.

**Cross-entropy (fused with softmax).** `loss = -mean log p_y`,
`dlogits = (softmax(z) - one_hot(y)) / B`, computed via the log-sum-exp
trick for stability.

**Conv2D.** im2col turns each receptive field into a column; the conv
becomes a GEMM `out = W_col @ cols`. Backward uses col2im to scatter
gradients, accumulating overlaps.

**LSTM (BPTT).** Gates `i, f, g, o`; `c_t = f*c_{t-1} + i*tanh(g)`;
`h_t = o*tanh(c_t)`. Backward walks time in reverse, carrying `dh_next`
and `dc_next`, and accumulates `dW`, `db` over all time steps.

**Adam.** First/second moments with bias correction
`m_hat = m/(1-beta1^t)`, `v_hat = v/(1-beta2^t)`;
`theta -= lr * m_hat / (sqrt(v_hat) + eps)`.

## Committed results (observed, not invented)

| Model | Task | Test accuracy | Notes |
|---|---|---|---|
| MLP | sklearn digits (8x8) | 0.9611 | 20 epochs, Adam |
| CNN | sklearn digits (8x8) | 0.9139 | 8 epochs, Adam |
| LSTM | parity of 8-bit sequence | 0.9875 | 50 epochs, Adam; loss 0.7362 -> 0.2011 |

Full configs are in `results/*.md`.

## Usage

```
python -m dlfs list
python -m dlfs check --layer dense|conv|lstm
python -m dlfs train --model mlp|cnn|lstm [--epochs N]
```

## Tests

```
python -m pytest tests -q
```

23 tests, all green: gradient checks, shape correctness, seeded determinism,
small-set overfit, Adam-vs-SGD smoke.

## Honest framing vs PyTorch

- Teaching-scale only. Pure Python/NumPy, no GPU, no autograd graph, no
  broadcasting magic beyond what is written.
- Digits is a toy benchmark; these numbers are not SOTA and not comparable
  to trained ResNets/ViTs.
- No model zoo, no pretrained weights, no production serving.

## Limitations

- Pure-Python loops (im2col, BPTT) are slow; large inputs are impractical.
- No GPU/CUDA.
- No automatic differentiation; gradients are hand-written per layer.
- Only the layers listed above; no attention, no normalization, no
  convTranspose, no bidirectional LSTM.
- Mini-batching and shuffling are basic; no data pipeline.
- Results are on small datasets and should not be quoted as benchmarks.

## Extensions

- Add batch normalization and dropout.
- Implement convTranspose and a U-Net.
- Bidirectional LSTM / GRU.
- CIFAR-10 with the existing Conv2D.
- A tiny autograd engine (Value/Tensor) for comparison.