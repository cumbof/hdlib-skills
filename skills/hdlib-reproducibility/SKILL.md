---
name: hdlib-reproducibility
description: Use when making hdlib runs deterministic - choosing seeds for Vector, Space.bulk_insert, ClassificationModel.fit / auto_tune / stepwise_regression, GraphModel, ClusteringModel, RegressionEncoder, RegressionModel, and the quantum models. Documents which APIs honour an explicit seed argument vs which use NumPy's global random state and therefore require np.random.seed before construction.
---

# Reproducibility in hdlib

`hdlib` mixes three reproducibility regimes:

1. **Explicit `seed=...` argument** &mdash; passed straight into
   `numpy.random.default_rng(seed)`. Fully deterministic.
2. **Global NumPy RNG** &mdash; the API uses `np.random.randn(...)`,
   `np.random.choice(...)`, etc. To make these deterministic, call
   `np.random.seed(N)` immediately before constructing the object.
3. **Pre-determined `random_state`** &mdash; sklearn primitives inside hdlib
   that use `random_state=0` (i.e. always reproducible).

This skill provides the **seed map**: what to set for each public class
and function.

## When to use this skill

- The user wants identical results across runs.
- The user is debugging "why does this give different distances each
  time?"
- The user is reviewing a CI test that checks numerical outputs.

## The seed map

| API | Reproducibility mechanism | Action |
|:----|:--------------------------|:-------|
| `Vector(...)` | Explicit `seed=int` argument | Pass `seed=...` |
| `Space(...)` | None &mdash; constructor does not generate random vectors | n/a |
| `Space.bulk_insert(names, tags=None)` | Internal `Vector(...)` calls have **no seed**; the iteration order over `names` is `set`-based (non-deterministic). Tags supplied positionally may even be misaligned. | Avoid `bulk_insert` when reproducibility matters. Build vectors with `Vector(seed=...)` and call `Space.insert(vec)` individually. |
| `ClassificationModel.fit(points, labels, seed=None)` | Explicit `seed` argument controls level vector RNG | Pass `seed=...` |
| `ClassificationModel.cross_val_predict` | `StratifiedKFold(random_state=0)` &mdash; deterministic split. Underlying `predict` uses the level vectors from `fit` | Seed `fit` once; the rest is deterministic |
| `ClassificationModel.auto_tune` | Each candidate calls `ClassificationModel(...).fit(...)` **without** a seed | Currently *not* deterministic. To make it so, copy and modify the implementation or seed before each fit yourself |
| `ClassificationModel.stepwise_regression` | Same as `auto_tune` &mdash; per-candidate `fit` is unseeded | Same workaround |
| `QuantumClassificationModel(seed=int, ...)` | Explicit `seed` argument | Pass `seed=...` |
| `QuantumClassificationModel.predict` (simulator) | `run_compute_uncompute_test(..., seed=self.seed)` -> `backend.run(..., seed_simulator=seed)` | Same as model seed |
| `QuantumClassificationModel.predict` (hardware) | Shot noise from real hardware; **not deterministic** | n/a (use simulator for reproducible tests) |
| `GraphModel(size, directed, seed=int)` | Explicit `seed`; controls only `self.rand` used for auto-threshold non-neighbour sampling. **Node and weight vectors are still constructed via `Vector(...)` without a seed,** so they are *not* deterministic across runs even with `seed=...` set. | Pass `seed=...` for threshold reproducibility, but expect node vectors to differ. To get fully deterministic graphs, pre-build the node `Vector`s with `Vector(seed=...)`, manually insert them into `graph.space`, then call `graph.fit(edges, build_nodes_memory=True)` |
| `GraphModel.edge_exists(..., threshold=None)` | Uses `self.rand` for non-neighbour sampling | Seeded by `GraphModel(seed=...)`. Or pass `threshold=...` explicitly |
| `ClusteringModel(k, n_features, seed=int)` | `np.random.seed(self.seed)` set in constructor. Projection matrix and initial centroids draw from the global RNG | Pass `seed=...`. **Does not reseed on subsequent fits** &mdash; construct a fresh model for each deterministic run |
| `RegressionEncoder(D, n_features)` | Uses global NumPy RNG (`np.random.randn`, `np.random.uniform`) | Call `np.random.seed(N)` immediately before constructing |
| `RegressionModel(...)` | `Vector(vtype="bipolar", size=D)` constructed for each cluster vector &mdash; no `seed` argument propagated | Call `np.random.seed(N)` immediately before constructing |
| `hdlib.arithmetic.quantum.encode(vec_bipolar)` | Pure function of the input | Deterministic given the input |
| `superposition_bundle`, `quantum_majority_bundle` | Pure functions | Deterministic |
| `run_compute_uncompute_test(... backend=AerSimulator(), seed=42)` | `seed_simulator=seed` for simulator | Pass `seed=...`. Hardware backends are inherently noisy |

## Recipes

### Vectors

```python
from hdlib.vector import Vector

v1 = Vector(name="v1", seed=1)        # always identical
v2 = Vector(name="v2", seed=1)        # same data as v1 (same seed!)
v3 = Vector(name="v3", seed=2)        # different
```

### Classification

```python
from hdlib.model import ClassificationModel

model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
model.fit(points, labels=labels, seed=0)        # deterministic level vectors

# cross_val_predict is automatically reproducible because of the
# random_state=0 in StratifiedKFold
results = model.cross_val_predict(points, labels, cv=5)
```

### Graphs

```python
from hdlib.model import GraphModel

graph = GraphModel(size=10000, directed=False, seed=0)
graph.fit(edges)
```

`seed=...` makes the **auto-threshold non-neighbour sampling** in
`edge_exists` deterministic. It does **not** seed the per-node
`Vector(name=..., size=...)` calls inside `_add_edge` &mdash; those still
draw from a fresh `np.random.default_rng()`. If you need the node and
weight vectors themselves to be bit-identical across runs, pre-construct
each node vector with `seed=...` and inject the `memory` / `weights`
attributes that `_add_edge` would normally attach:

```python
import numpy as np
from hdlib.vector import Vector

graph = GraphModel(size=10000, directed=False, seed=0)
for node in {n for edge in edges for n in edge[:2]}:
    v = Vector(name=node, size=graph.size, vtype=graph.vtype,
               seed=hash(node) % (2**31))
    setattr(v, "memory", None)
    setattr(v, "weights", {})
    graph.space.insert(v)
graph.fit(edges)
```

### Clustering

```python
import numpy as np
from hdlib.model import ClusteringModel

# ClusteringModel(seed=...) reseeds the global NumPy RNG once.
# For full determinism, also disallow other library calls between
# construction and fit.
model = ClusteringModel(k=4, n_features=8, size=10000, seed=0)
model.fit(X)
```

If you need to refit on the same data and expect identical labels each
time, construct a **new** model:

```python
model_a = ClusteringModel(k=4, n_features=8, size=10000, seed=0)
model_a.fit(X)

model_b = ClusteringModel(k=4, n_features=8, size=10000, seed=0)
model_b.fit(X)

assert (model_a.labels_ == model_b.labels_).all()    # always
```

### Regression

```python
import numpy as np
from hdlib.model.regression import RegressionModel

np.random.seed(0)                # IMPORTANT: before construction
model = RegressionModel(D=10000, n_features=8, k_models=8, iterations=20)
model.fit(X_train, y_train)
```

Calling `np.random.seed(...)` *after* the constructor has no effect on
the encoder or cluster vectors that were already drawn from the global
RNG inside `__init__`.

### Quantum classification

```python
from hdlib.model import QuantumClassificationModel

model = QuantumClassificationModel(size=32, levels=4, seed=42, shots=1024)
model.fit(X, y)
preds, _ = model.predict(X_test)        # simulator -> deterministic
```

The `seed` propagates into both the level vector generator and
`run_compute_uncompute_test(..., seed=self.seed)`. Predictions on
simulator are bit-exact across runs.

## What is *not* currently reproducible (and why)

- **`auto_tune` / `stepwise_regression`** &mdash; the helper
  `_init_fit_predict` constructs each candidate model with
  `ClassificationModel(...).fit(points, labels=labels)` *without*
  passing a seed. Different runs produce different level vectors and
  therefore different cross-validated scores. To work around this:
  fork the code or call `_init_fit_predict` directly and pass `seed=...`
  to `model.fit`.

- **Real-hardware quantum runs** &mdash; shot noise and time-varying noise
  on IBM backends are intrinsic; the model has no way to reproduce
  results across submissions.

- **`Space.bulk_insert`** &mdash; the generated `Vector` objects inside
  are constructed without a `seed`. Build the vectors manually with
  `Vector(seed=...)` and `Space.insert(...)` for deterministic spaces.

- **`RegressionEncoder` and `RegressionModel`'s cluster vectors** &mdash;
  use the global RNG. Wrap with `np.random.seed(...)` immediately
  before construction.

## Cross-cutting recipe &mdash; deterministic pipeline

```python
import numpy as np
from hdlib.model import ClassificationModel

# Top-of-script seed for everything that uses the global numpy RNG
np.random.seed(0)

# Explicit seeds for everything that accepts one
model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
model.fit(points, labels=labels, seed=0)

# Use cross-validation (already deterministic) instead of auto_tune
results = model.cross_val_predict(points, labels, cv=5, retrain=10, n_jobs=1)
```

Running this script twice should give bit-identical predictions and
distances.

## Common pitfalls

- **Forgetting `np.random.seed(...)` before constructing `RegressionEncoder`.**
  The encoder is allocated in `__init__` and uses the global RNG &mdash;
  there is no second chance to seed it.
- **Calling `np.random.seed(...)` between `__init__` and `fit`.** The
  bytes that were already drawn in `__init__` are committed; only future
  draws are affected.
- **Using `n_jobs > 1` with a seed.** Multiprocessing workers inherit
  the parent's RNG state, but their forks/spawns may diverge slightly
  on macOS spawn. Stick to `n_jobs=1` for bit-exact reproducibility.
- **Mixing seeded `Vector` construction with `Space.bulk_insert`.**
  `bulk_insert` rebuilds the vectors internally without seeds. Choose
  one strategy and stick to it.
- **Comparing pickle round-trips across hdlib versions.** Even with
  matching seeds, internal attributes added in newer versions will not
  be present on older-version pickles, causing subtle behaviour
  differences. Pin `hdlib==X.Y.Z`.

## See also

- `hdlib-vectors` and `hdlib-space` for the `seed` argument on `Vector(...)`.
- `hdlib-classification` for the `seed` argument on `fit`.
- `hdlib-graph`, `hdlib-clustering`, `hdlib-regression` for their
  seeding quirks.
- `hdlib-pitfalls` (item "Non-deterministic runs") for a one-paragraph
  summary.
