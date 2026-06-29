---
name: hdlib-hyperparameter-tuning
description: Use when sweeping the hyperdimensional vector size and number of level vectors for an hdlib ClassificationModel via the auto_tune method - a Parameter Sweep Analysis that cross-validates every (size, levels) pair in the supplied ranges. Covers parameters (size_range, levels_range, cv, distance_method, retrain, n_jobs, metric), return value, and tie-breaking rules.
---

# Hyperparameter sweep with `ClassificationModel.auto_tune`

`auto_tune` performs a **Parameter Sweep Analysis** (PSA) over a Cartesian
product of vector sizes and level counts, picks the pair with the highest
cross-validated score, and breaks ties by preferring **fewer levels** and
then **smaller size**.

## When to use this skill

- The user wants to find a good `(size, levels)` configuration for a
  `ClassificationModel`.
- The user calls `ClassificationModel.auto_tune(...)`.
- The user is benchmarking HDC at different dimensionalities.

## Signature

```python
ClassificationModel.auto_tune(
    points: list[list[float]],
    labels: list[str],
    size_range: range,                  # e.g. range(1000, 10001, 1000)
    levels_range: range,                # e.g. range(2, 21, 2)
    cv: int = 5,
    distance_method: str = "cosine",
    retrain: int = 0,
    n_jobs: int = 1,
    metric: str = "accuracy",           # "accuracy" | "f1" | "precision" | "recall"
) -> tuple[int, int, float]:
    """ -> (best_size, best_levels, best_metric) """
```

The ranges are standard Python `range` objects; the function evaluates
every `(size, levels)` pair where:

- `size > len(points)` (otherwise the candidate is skipped &mdash; the
  classifier needs more space dimensions than samples).
- `levels > 1`.

## Algorithm

For each `(size, levels)` candidate:

1. Build a fresh `ClassificationModel(size, levels, vtype=self.vtype)`.
2. Run `.fit(points, labels)`.
3. Run `.cross_val_predict(...)` with the supplied `cv`, `distance_method`,
   `retrain`, and `n_jobs=1` (parallelism happens at the candidate level
   instead).
4. Compute the mean cross-validated score using `metric`.

Then choose the candidate with the highest score. On a tie, prefer the
one with fewer `levels`; on a further tie, prefer smaller `size`. Returns
`(best_size, best_levels, best_metric)`.

## Pre-conditions / errors

| Condition | Exception |
|:----------|:----------|
| `points` empty | `Exception` |
| `labels` empty | `Exception` |
| `len(points) != len(labels)` | `ValueError` |
| `cv < 2` or `cv > len(points)` | `ValueError` |
| `retrain < 0` | `ValueError` |
| `metric` not in supported set | `ValueError` (from `_init_fit_predict`) |

## Worked example

```python
from sklearn.datasets import load_wine
from hdlib.model import ClassificationModel

wine = load_wine()
points = wine.data.tolist()
labels = wine.target.tolist()

model = ClassificationModel()  # only used as a holder for vtype = "bipolar"
best_size, best_levels, best_acc = model.auto_tune(
    points=points,
    labels=labels,
    size_range=range(2000, 10001, 2000),    # 2k, 4k, 6k, 8k, 10k
    levels_range=range(5, 21, 5),           # 5, 10, 15, 20
    cv=5,
    metric="accuracy",
    n_jobs=4,
)

print(f"best (size, levels, accuracy) = ({best_size}, {best_levels}, {best_acc:.4f})")
```

The output is a single best candidate, not a ranking. If you need the
full surface, copy `auto_tune`'s implementation and store every
`(size, levels, score)` triple.

## Compute cost

`auto_tune` evaluates `len(size_range) * len(levels_range)` candidates,
each of which trains and cross-validates a `ClassificationModel`. Each
candidate is a 5-fold cross-validation by default. With `n_jobs=4`
(processes), candidates run in parallel.

For Iris-scale data (150 samples, 4 features) on a modern laptop:
- 5x5 grid (`size_range` of 5 values, `levels_range` of 5 values) is on
  the order of 30 - 60 seconds with `n_jobs=4`.
- 10x10 grid is roughly 5 - 10 minutes.

For tabular datasets with thousands of samples, each candidate can take
tens of seconds; budget accordingly.

## Tying with retraining

`retrain` is forwarded to every candidate's cross-validated `predict`.
Higher `retrain` increases the per-candidate cost roughly linearly with
the actual number of retraining iterations performed (each iteration
recomputes the training error rate). For large grids, start with
`retrain=0`, find the best `(size, levels)`, then re-evaluate with
`retrain=10..20` on the winning candidate.

## Choosing a sensible range

- `size_range`: start at `max(1000, len(points) + 1)`. For most tabular
  problems, sizes above `10000` see diminishing returns.
- `levels_range`: start at 2. For continuous features with broad ranges,
  10 - 20 levels usually suffice; finer-grained levels may hurt
  generalisation.
- Pick step sizes coarse enough to keep the grid small at first
  (e.g. `range(2000, 10001, 2000)`); refine around the best candidate
  with a finer grid afterwards.

## What `auto_tune` does *not* tune

- `vtype` &mdash; you must construct the receiver model with the desired
  `vtype` (the candidate models inherit it).
- `distance_method` &mdash; same as `vtype`, supplied to every candidate.
- Feature selection &mdash; use `stepwise_regression` separately.

## Common pitfalls

- **`size <= len(points)` candidates are silently skipped.** If every
  candidate falls into this case, `auto_tune` returns `(None, None, None)`.
  Ensure `min(size_range) > len(points)`.
- **`n_jobs > 1` forks the entire interpreter** (because hdlib uses
  `multiprocessing.Pool`). On macOS notebooks this works only when the
  call is inside `if __name__ == "__main__":`.
- **Stochasticity.** Each candidate uses fresh random level vectors and
  Stratified K-Fold splits with `random_state=0`. The level vectors are
  *not* seeded; results vary across runs. Run `auto_tune` 3 - 5 times
  if you need a confident best.

## See also

- `hdlib-classification` &mdash; the model `auto_tune` is tuning.
- `hdlib-feature-selection` &mdash; complementary procedure for picking
  features at fixed `(size, levels)`.
- `hdlib-reproducibility` &mdash; current `auto_tune` is non-deterministic
  beyond the fold split; document that limitation.
