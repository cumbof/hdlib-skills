---
name: hdlib-classification
description: Use when building, fitting, predicting with, cross-validating, or retraining a supervised classifier on tabular numerical data using hdlib's hyperdimensional computing approach. Covers the hdlib.model.classification.ClassificationModel class - constructor (size, levels, vtype), fit (level vectors and point encoding), predict (with retraining), cross_val_predict, error_rate, and the underlying _encode_point logic. Load whenever the user creates a ClassificationModel, encodes data points into level-based vectors, or implements the Cumbo et al. 2020 supervised HDC algorithm.
---

# Supervised classification with `ClassificationModel`

`hdlib.model.classification.ClassificationModel` implements the supervised
HDC classifier described in
[Cumbo, Cappelli & Weitschek 2020](https://doi.org/10.3390/a13090233):
each numerical feature value is quantised to one of `levels` random
**level vectors**, those level vectors are **permuted** by the feature
index, then **bundled** into one point vector. A class prototype is the
bundle of every training point of that class. Inference is nearest-class
on cosine distance.

## When to use this skill

- The user constructs `ClassificationModel(...)`, calls `.fit`, `.predict`,
  `.cross_val_predict`, `.error_rate`, or `.stepwise_regression`.
- The user has a tabular dataset (each row a sample, last column a class
  label) and wants to train a classifier with HDC.
- The user needs to retrain the model on misclassified samples.

For the **quantum** classifier see `hdlib-quantum-classification`. For
**feature ranking** see `hdlib-feature-selection`. For **hyperparameter
tuning** see `hdlib-hyperparameter-tuning`.

## Import

```python
from hdlib.model import ClassificationModel
# or, equivalently:
from hdlib.model.classification import ClassificationModel
```

## Constructor

```python
ClassificationModel(
    size: int = 10000,           # vector dimensionality
    levels: int = 2,             # number of level vectors (>= 2)
    vtype: str = "bipolar",      # "binary" or "bipolar"
)
```

- `size` must be an `int`. Convention: a few thousand to tens of
  thousands. Smaller sizes hurt accuracy; larger sizes increase memory.
- `levels` must be `>= 2`. Trade-off:
  - Fewer levels => coarser quantisation => more aggressive generalisation.
  - Many levels => fine-grained quantisation but fewer training samples
    per level region => potential overfitting.
- `vtype` defaults to `"bipolar"` (best for cosine distance).

The constructor only stores configuration; the actual `Space` and level
vectors are created in `fit`. Attributes after `__init__`:

| Attribute | Type | Notes |
|:----------|:-----|:------|
| `size`, `levels`, `vtype` | as input |  |
| `min_value`, `max_value` | `float | None` | filled in by `fit` |
| `level_list` | `list[(float, float)]` | per-level `(left, right)` bin bounds |
| `space` | `Space | None` | the underlying space |
| `classes` | `set` | empty until `fit` |
| `version` | `str` | hdlib version |

## End-to-end recipe (3 lines of substance)

```python
from sklearn.datasets import load_iris
from hdlib.model import ClassificationModel

iris = load_iris()
points = iris.data.tolist()
labels = iris.target.tolist()

model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
model.fit(points, labels=labels, seed=42)
results = model.cross_val_predict(points, labels, cv=5, retrain=10, n_jobs=1)
```

`results` is a list of 5 tuples (one per fold):
`(test_indices, predictions, distances, retraining_iters, error_rate, class_vectors)`.

## `fit(points, labels, seed=None)`

Encodes every training sample into the space.

- `points: list[list[float]]` &mdash; one inner list per sample (numerical).
- `labels: list[str]` &mdash; same length as `points`.
- `seed: int | None` &mdash; controls level-vector generation.

### Pre-conditions

- `len(points) >= 3` (else `Exception("Not enough data points")`).
- `len(points) == len(labels)`.
- `len(set(labels)) >= 2`.

### What `fit` does internally

1. Discovers `min_value` and `max_value` across all features (across all
   samples), divides that range into `levels` equal bins, and records
   `(left, right)` for each bin in `self.level_list`.
2. Generates `levels` random vectors `level_0, level_1, ...` such that
   consecutive levels differ by approximately `D / (2 * levels)` positions
   &mdash; this is the **similarity-preserving** level encoding from the
   paper. The first level differs from the second by `D/2` positions
   (orthogonal-ish), each subsequent one is closer to its predecessor.
3. Stores those level vectors in `self.space` with names `"level_0"`,
   `"level_1"`, ...
4. Calls `_encode_point` on each sample:
   - For each feature value, find the matching bin index.
   - Look up `level_<i>` and permute it by the feature column index.
   - Bundle all permuted level vectors together.
5. Adds the encoded point to the space as `point_<n>` and tags it with
   the class label.

After `fit`, the space contains `levels + len(points)` vectors.

## `predict(test_indices, distance_method="cosine", retrain=0)`

Holds out the indices in `test_indices` and trains on every other point.
Returns the rich tuple:

```python
test_indices, predictions, distances, retraining_iters, error_rate, class_vectors
```

- **`predictions`** &mdash; list of class labels, same order as `test_indices`.
- **`distances`** &mdash; for each test sample, a list of distances against
  each class vector (in the order `self.classes` was iterated).
- **`retraining_iters`** &mdash; how many retraining passes actually ran
  (may be less than `retrain` if the error rate stopped improving).
- **`error_rate`** &mdash; final training error rate **after** retraining.
- **`class_vectors`** &mdash; the (possibly retrained) class prototype vectors.

### Retraining loop

`retrain > 0` enables iterative error mitigation:

1. Build class prototypes (bundle of training points per class).
2. Compute training error rate (`error_rate`).
3. For each misclassified sample, add it to the correct class prototype
   and subtract it from the wrongly-predicted prototype.
4. Recompute error rate. If it got worse, revert to the previous
   iteration's prototypes and stop.
5. Otherwise repeat up to `retrain` times.

### Pre-conditions / errors

- `test_indices` must be non-empty (`Exception`).
- `retrain >= 0` (else `ValueError`).
- `len(self.classes) > 0` (i.e. `fit` was called first).

### Inferring class label order

`self.classes` is a Python `set`, so its iteration order is insertion
order in CPython 3.7+ but logically unordered. To map `distances[i]`
back to class labels reliably, sort:

```python
class_labels = sorted(self.classes)
```

or read each class vector's tag from `class_vectors`:

```python
order = [list(cv.tags)[0] for cv in class_vectors]
```

## `cross_val_predict(points, labels, cv=5, distance_method="cosine", retrain=0, n_jobs=1)`

Stratified k-fold cross-validation. Returns a list of `predict` tuples
(one per fold). The fold split is deterministic
(`StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)`).

- `cv >= 2` and `cv <= len(points)` (else `ValueError`).
- `retrain >= 0` (else `ValueError`).
- `n_jobs < 1` is interpreted as "all available CPUs" via `os.cpu_count()`.
- When `n_jobs > 1`, folds run in parallel via `multiprocessing.Pool`.

### Computing fold-level metrics

```python
from sklearn.metrics import accuracy_score

for test_indices, preds, _, _, _, _ in results:
    y_true = [labels[i] for i in test_indices]
    print(accuracy_score(y_true, preds))
```

## `error_rate(training_vectors, class_vectors, distance_method="cosine")`

Used internally by `predict`'s retraining loop. Computes:

- Fraction of `training_vectors` misclassified by nearest-class on
  `class_vectors`.
- The list of misclassified `Vector` objects and the wrong class label
  predicted for each.

Useful if you've stripped the model from the space and want to evaluate it
externally; otherwise prefer `predict`.

## Internal helpers (private but worth knowing)

- `_encode_point(point)` &mdash; pure function that encodes a single sample
  using the current `level_list` and the level vectors in `self.space`.
  Sums (`bundle`s) permuted level vectors across feature indices.
- `_init_fit_predict(...)` &mdash; throwaway model builder used by
  `auto_tune` and `stepwise_regression` to evaluate a candidate
  `(size, levels)` or feature subset.
- `_stepwise_regression_iter(...)` &mdash; runs one step of forward / backward
  feature elimination. See `hdlib-feature-selection`.

## Worked example &mdash; train, retrain, evaluate

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from hdlib.model import ClassificationModel

# 1. data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X.tolist(), y.tolist(), test_size=0.2, random_state=0, stratify=y
)
# Combine train + test back into single arrays
points = X_train + X_test
labels = y_train + y_test
test_indices = list(range(len(X_train), len(points)))

# 2. fit
model = ClassificationModel(size=10000, levels=20, vtype="bipolar")
model.fit(points, labels=labels, seed=0)

# 3. predict (with up to 20 retraining iters)
test_idx, preds, _, n_iters, err, class_vectors = model.predict(
    test_indices=test_indices, retrain=20
)
print(f"retraining iters: {n_iters}; final training error: {err}")
print("accuracy:", accuracy_score(y_test, preds))
print("f1:      ", f1_score(y_test, preds, average="weighted"))
```

## Common pitfalls

- **`levels` too small.** With `levels=2` the encoder collapses every
  feature value into one of just two bins. Use `levels >= 10` for most
  real datasets.
- **`size` too small.** With `size < 1000`, level vectors collide and the
  model degenerates. Use `size >= 5000` unless you're explicitly
  benchmarking small dimensions.
- **Mixing numeric scales** &mdash; the model bins every feature against a
  *single* `[min_value, max_value]` range across **all** features. If
  features have wildly different scales, the smaller-range features are
  collapsed into one or two bins. Standardise / min-max-scale your data
  before `fit`.
- **String features.** `_encode_point` calls comparisons like `value == self.min_value`
  and `if left_bound <= value and right_bound > value`. Non-numeric input
  silently produces an exception in NumPy or yields invalid bins. One-hot
  encode or numerically encode categorical features first.
- **Re-`fit`ing on the same model** overwrites the space &mdash; any vectors
  you inserted manually are lost.
- **`predict` with `test_indices` that include non-existent indices** &mdash;
  raises `Exception("Unable to retrieve all the test vectors from the space")`.
- **`n_jobs > 1` on small datasets** is slower than `n_jobs=1` due to
  process startup cost; use parallelism only for `cv >= 5` and
  `len(points) >= 1000`.
- **Class label order is set-based** &mdash; `self.classes` is a `set`; do
  not assume positional alignment with anything else. Use `sorted(self.classes)`
  or inspect `class_vectors[i].tags`.

## See also

- `hdlib-feature-selection` &mdash; `model.stepwise_regression` for HDC-based
  feature ranking.
- `hdlib-hyperparameter-tuning` &mdash; `model.auto_tune` for sweeping
  `(size, levels)`.
- `hdlib-encoding-data` &mdash; deeper discussion of the level-vector
  encoding and alternatives.
- `hdlib-quantum-classification` &mdash; the Qiskit-backed variant.
- `hdlib-pitfalls` &mdash; common runtime errors during `fit` / `predict`.
