---
name: hdlib-regression
description: Use when building a regression model on numerical data with hdlib's RegHD approach via hdlib.model.regression. Covers both classes - RegressionEncoder (the cos*sin similarity-preserving encoding) and RegressionModel (multi-model RegHD with clustering and per-cluster regression heads), with the constructor parameters (D, n_features, k_models, learning_rate, iterations, binary_threshold), fit / predict / set_quantized_prediction_mode, the iterative update rule, and the softmax-weighted ensemble inference.
---

# RegHD regression with `RegressionEncoder` and `RegressionModel`

`hdlib.model.regression` implements the **RegHD** multi-model regression
algorithm described in
[Hernandez-Cano et al. 2021](https://doi.org/10.1109/DAC18074.2021.9586284).
It uses a `cos * sin` random-basis encoder, a bank of `k` cluster
hypervectors that act as soft assignments, and a parallel bank of `k`
real-valued regression hypervectors. Inference is a softmax-weighted dot
product across the bank.

## When to use this skill

- The user wants to predict a continuous target with HDC.
- The user instantiates `RegressionEncoder(...)` or `RegressionModel(...)`.
- The user calls `.fit(X, y)` / `.predict(X_query)` / `.set_quantized_prediction_mode(...)`.

## Imports

```python
from hdlib.model.regression import RegressionEncoder, RegressionModel
# Also reexported from the top-level model package:
from hdlib.model import RegressionEncoder, RegressionModel
```

## `RegressionEncoder`

### Purpose

Maps an `n_features`-dimensional float row to a `D`-dimensional **real**
hypervector via:

```
h_i = cos(F . B_i + b_i) * sin(F . B_i)
```

where `B_i in R^D` are random Gaussian base vectors (one per feature) and
`b_i ~ U(0, 2pi)` is a random bias vector. Each `B_i` corresponds to a
**feature**, and the encoder is similarity-preserving: nearby inputs
yield close hypervectors.

### Constructor

```python
RegressionEncoder(D: int, n_features: int)
```

- `D > 0`, `n_features > 0` (both must be ints; else `ValueError`).
- Allocates `base_hypervectors: (n_features, D)` from `np.random.randn`
  and `biases: (D,)` from `U(0, 2pi)`.

> The encoder uses NumPy's **global** RNG. For reproducibility, call
> `np.random.seed(...)` immediately before constructing.

### `encode(feature_vector)`

```python
encoded = encoder.encode(feature_vector)
```

- `feature_vector` must have `n_features` entries (raises `ValueError`).
- Returns a 1-D `np.ndarray` of shape `(D,)` with real values.

The output is **not** bipolar; the downstream `RegressionModel`
binarises it via `_to_bipolar`.

## `RegressionModel`

### Constructor

```python
RegressionModel(
    D: int,                       # HD dimensionality
    n_features: int,
    k_models: int = 8,            # number of soft clusters / regression heads
    learning_rate: float = 0.01,
    iterations: int = 20,
    binary_threshold: float = 0.0,
)
```

- All numeric arguments must be positive (`D, n_features, k_models` are
  positive ints; `learning_rate, iterations` positive; `binary_threshold`
  any number).
- Allocates internal state:
  - `cluster_models_b: list[Vector]` &mdash; `k` random bipolar Vectors used
    for *soft cluster assignment*.
  - `cluster_models_int: list[np.ndarray]` &mdash; their float versions for
    incremental updates.
  - `regression_models_int: list[np.ndarray]` &mdash; `k` zero-initialised
    real vectors of size `D` &mdash; the regression heads.
  - `regression_models_b: list[Vector]` &mdash; bipolar copies for
    optional quantized prediction.
  - `encoder: RegressionEncoder(D, n_features)`.

### `fit(X, y)`

```python
model.fit(X, y)
```

- `X: np.ndarray, shape (n_samples, n_features)`.
- `y: np.ndarray, shape (n_samples,)`.
- Returns `None`. Prints per-epoch MSE to stdout.

#### Per-epoch loop (simplified)

```python
for iteration in range(self.iterations):
    for i in range(n_samples):
        encoded_float  = encoder.encode(X[i])
        encoded_b      = self._to_bipolar(encoded_float)

        # Soft cluster assignment
        sims    = [1 - encoded_b.dist(c, method="cosine") for c in cluster_models_b]
        weights = softmax(sims)

        # Soft prediction
        y_hat = sum(w * dot(r_int, encoded_float) for w, r_int in zip(weights, regression_models_int))
        err   = y[i] - y_hat

        # Update regression heads
        for j in range(k_models):
            regression_models_int[j] += lr * weights[j] * err * encoded_float

        # Update the cluster the sample belongs to (max similarity)
        l_max = argmax(sims)
        cluster_models_int[l_max] += (1 - weights[l_max]) * encoded_float

    # Quantise cluster_models_b at the end of each epoch
    for j in range(k_models):
        cluster_models_b[j] = self._to_bipolar(cluster_models_int[j])
```

The model prints MSE every epoch. After `iterations` epochs the regression
heads are ready for inference.

### `predict(X_query)`

```python
y_pred = model.predict(X_query)        # shape: (X_query.shape[0],)
```

For each query sample:

1. Encode (`encoded_float`, `encoded_b`).
2. Compute softmax weights from cosine similarities against
   `cluster_models_b`.
3. Sum `weights[j] * dot(regression_head[j], encoded_float)` across heads.
4. Return the scalar prediction.

If `set_quantized_prediction_mode(True)` was called, the dot product is
computed in **bipolar** space instead (`regression_models_b[j].vector`
vs `encoded_b.vector`).

### `set_quantized_prediction_mode(enable=True)`

Toggles between:

- **Float prediction (default).** Uses `regression_models_int` (real
  vectors). Higher accuracy, more memory.
- **Quantised prediction.** Uses `regression_models_b` (bipolar
  `Vector`s). Faster inference, lower memory, slightly lower accuracy.

Calling with `enable=True` also re-binarises every regression head from
its current `_int` state.

## Worked example &mdash; predict a continuous target

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from hdlib.model.regression import RegressionModel

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reproducible encoder
np.random.seed(0)

model = RegressionModel(
    D=10000,
    n_features=X.shape[1],
    k_models=8,
    learning_rate=0.005,
    iterations=20,
    binary_threshold=0.0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))
```

For `D=10000` and `iterations=20` on a modern laptop this takes a few
minutes. RegHD trades training cost for inference speed; the predictor
becomes a handful of dot products.

## Tips for good results

- **Scale `y`** to a unit range before `fit`. The training loop applies a
  raw `learning_rate * error * encoded_float` update; large `y` magnitudes
  push the regression heads quickly past stability.
- **Standardise / min-max scale `X`.** The encoder is sensitive to feature
  magnitudes (they enter as the argument of `cos` and `sin`); features
  with magnitudes much larger than 1 wrap around the trig functions.
- **`k_models` controls capacity.** More heads means more local
  flexibility but slower convergence. Start with `k_models=8` and grid
  search up to 32.
- **`learning_rate` ~ 0.005 - 0.05** typically works. Inspect the printed
  MSE trajectory; if it explodes, halve the rate.
- **`iterations`** matters less than learning rate &mdash; a small rate with
  more iterations beats a high rate with few.

## Common pitfalls

- **Non-numpy inputs.** Both `fit` and `predict` start with
  `np.asarray(..., dtype=float)`, so lists work, but `pandas.DataFrame`
  also works only because asarray succeeds on it. Convert explicitly to
  keep your code clear.
- **Shape errors.** `X.shape[1] != n_features` raises `ValueError`. Pass
  the correct `n_features` to the constructor; do not rely on auto-detection.
- **Reproducibility.** Both `RegressionEncoder` and `RegressionModel`'s
  cluster vectors use the global NumPy RNG. To get deterministic runs,
  call `np.random.seed(...)` immediately before constructing the
  encoder/model.
- **Quantised prediction quality.** Switching to quantised mode after
  training can drop R^2 noticeably; only enable it when latency is more
  important than the last few accuracy points.
- **Output goes to stdout.** Use `contextlib.redirect_stdout` to silence
  the per-iteration MSE prints during long sweeps.

## See also

- `hdlib-vectors` &mdash; both float and bipolar encodings round-trip through
  `Vector`.
- `hdlib-distance` &mdash; cosine drives the cluster assignment.
- `hdlib-classification` &mdash; supervised counterpart for categorical
  targets.
- `hdlib-encoding-data` &mdash; the cos*sin encoder is one of several
  encoders covered in that pattern skill.
