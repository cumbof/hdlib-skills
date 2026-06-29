---
name: hdlib-clustering
description: Use when performing unsupervised clustering on numerical data with hdlib's hyperdimensional k-means via hdlib.model.clustering.ClusteringModel. Covers constructor (k, n_features, size, vtype, max_iter, seed), the projection-matrix encoder, the fit/predict cycle, convergence detection, and the difference between hdlib clustering and classical k-means.
---

# `ClusteringModel` &mdash; HDC-based k-means

`hdlib.model.clustering.ClusteringModel` projects raw numerical data into
a high-dimensional bipolar space via a Gaussian random projection,
binarises with `np.sign`, and then runs k-means in HD space using cosine
distance. Based on
[Gupta et al. 2022](https://doi.org/10.1145/3503541).

## When to use this skill

- The user wants to cluster numerical data unsupervised with HDC.
- The user calls `ClusteringModel(...)`, `.fit(X)`, or `.predict(X)`.

## Import

```python
from hdlib.model import ClusteringModel
# or
from hdlib.model.clustering import ClusteringModel
```

## Constructor

```python
ClusteringModel(
    k: int,                         # number of clusters, required
    n_features: int,                # number of input features, required
    size: int = 10000,              # HD vector dimensionality
    vtype: str = "bipolar",         # only "bipolar" is meaningful here
    max_iter: int = 100,
    seed: Optional[int] = None,
)
```

### Pre-conditions

- `k > 0` (else `ValueError`).
- `n_features > 0` (else `ValueError`).
- `size` must be an int (else `ValueError`).
- `max_iter > 0` (else `ValueError`).
- `seed` must be `int | None` (else `TypeError`).

The constructor seeds NumPy's global RNG via `np.random.seed(self.seed)`
(only when `seed is not None`) and immediately allocates the projection
matrix:

```python
self.projection_matrix_ = np.random.randn(self.n_features, self.size)
```

That matrix is the **encoder**: any `n_features`-dimensional input row
becomes a `size`-dimensional float vector via `X @ projection_matrix_`,
which is then `np.sign(...)`'d to bipolar.

## Attributes after `__init__`

| Attribute | Type | Notes |
|:----------|:-----|:------|
| `projection_matrix_` | `np.ndarray (n_features, size)` | Gaussian-random encoder |
| `centroids_` | `list[Vector]` | empty until `fit` |
| `labels_` | `np.ndarray` | empty until `fit` |

## `fit(X)` and `predict(X)`

```python
def fit(X: np.ndarray) -> "ClusteringModel": ...
def predict(X: np.ndarray) -> np.ndarray: ...
```

### Pre-conditions on `X`

- `X` must be a 2-D `numpy.ndarray` (`TypeError` otherwise).
- `X.shape[1] == n_features` (else `ValueError`).

### `fit` algorithm

1. Encode every row of `X` into a bipolar `Vector` via the projection
   matrix + `np.sign`.
2. Pick `k` initial centroids at random (without replacement) from the
   encoded points.
3. Loop up to `max_iter` times:
   a. Assign each point to the nearest centroid using **cosine** distance.
   b. Recompute each centroid as the bundle of its assigned points
      divided by the cluster size (this divides the raw bipolar sum, so
      centroids contain non-bipolar values).
   c. If labels did not change from the previous iteration, print a
      "Converged" message and break.
4. Store the final `centroids_` and `labels_`.

> **Note:** the recomputed centroid is not normalised back to bipolar.
> Subsequent cosine-distance comparisons work because cosine is
> magnitude-invariant.

### `predict` algorithm

1. Encode `X` the same way `fit` did.
2. For each encoded point, return `argmin` of cosine distance to each
   centroid in `centroids_`.

Raises `RuntimeError` if `predict` is called before `fit`.

## Worked example

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from hdlib.model import ClusteringModel

X, y = make_blobs(n_samples=300, centers=4, n_features=8, random_state=0)

model = ClusteringModel(k=4, n_features=X.shape[1], size=10000, seed=0)
model.fit(X)
pred = model.predict(X)

print("ARI:", adjusted_rand_score(y, pred))
print("Cluster sizes:", np.bincount(model.labels_))
```

On synthetic blobs you should see ARI above 0.8 most of the time.

## Difference vs scikit-learn k-means

- **No deterministic init.** Centroids are sampled randomly from the
  encoded points. Pass `seed` for reproducibility &mdash; but note that the
  seed only affects the **first** call; subsequent `fit`s on the same
  instance reuse the projection matrix and rely on global NumPy random
  state for centroid initialisation.
- **Distance metric is cosine** by construction. Cluster shapes are
  unbiased by point magnitude in raw feature space (the encoder
  effectively performs a random projection).
- **Output is a label vector**, just like sklearn. Inspect
  `model.centroids_` if you need the actual centroid hypervectors.

## Common pitfalls

- **`X` must be a `numpy.ndarray`.** A pandas DataFrame or list of lists
  raises `TypeError`. Convert with `df.to_numpy()` or `np.asarray(...)`.
- **Re-seeding for repeated `fit` calls** does not re-randomise the
  projection matrix (it is built in `__init__`). For fully fresh runs,
  construct a new `ClusteringModel`.
- **Centroids are not bipolar after `fit`.** They hold float values from
  the per-cluster mean. Do not feed them back through `Vector(vtype="bipolar")`
  expecting strict `{-1, +1}` values.
- **Cluster `k` must be `<= n_samples`.** Otherwise the random
  `np.random.choice(num_points, size=k, replace=False)` raises
  `ValueError`. Validate this in the caller.
- **Empty clusters.** If a cluster receives no points in an iteration,
  the code falls back to a random point as the new centroid. Repeated
  empty clusters can cause oscillation rather than convergence.
- **Convergence message goes to stdout.** Wrap `fit` with
  `contextlib.redirect_stdout` if you want silent runs.

## See also

- `hdlib-vectors` and `hdlib-distance` &mdash; centroids are `Vector`s and
  distances are cosine.
- `hdlib-classification` &mdash; the supervised counterpart.
- `hdlib-regression` &mdash; uses the same Gaussian-projection idea on the
  encoder side.
- `hdlib-reproducibility` &mdash; `ClusteringModel(seed=...)` only affects
  the **first** fit call.
