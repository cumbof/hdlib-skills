---
name: hdlib-feature-selection
description: Use when ranking features by importance with hdlib's hyperdimensional computing approach via the ClassificationModel.stepwise_regression method - either backward variable elimination or forward variable selection. Covers parameters (method, cv, distance_method, retrain, metric, threshold, uncertainty, stop_if_worse), the iterative algorithm, return shape (importances, scores, best_importance, count_models), and parallel execution.
---

# HDC-based feature selection with `stepwise_regression`

`ClassificationModel.stepwise_regression` is a wrapper that uses the
hyperdimensional classifier itself as the underlying scorer for stepwise
feature selection. It rebuilds and cross-validates a fresh classifier for
each candidate feature subset, then ranks features by how often they
appear in subsets that hit the best score.

## When to use this skill

- The user wants to rank features by importance using HDC.
- The user calls `ClassificationModel.stepwise_regression(...)`.
- The user has a labelled tabular dataset and wants to drop irrelevant
  features.

## Signature

```python
ClassificationModel.stepwise_regression(
    points: list[list[float]],
    features: list[str],
    labels: list[str],
    method: str = "backward",          # "backward" or "forward"
    cv: int = 5,
    distance_method: str = "cosine",
    retrain: int = 0,
    n_jobs: int = 1,
    metric: str = "accuracy",          # "accuracy" | "f1" | "precision" | "recall"
    threshold: float = 0.6,            # stop if best score < threshold
    uncertainty: float = 5.0,          # percent tolerance for "tied" subsets
    stop_if_worse: bool = False,       # stop if iteration score drops below previous - tolerance
) -> tuple[dict[str, int], dict[int, float], int, int]:
    """ -> (importances, scores, best_importance, count_models) """
```

## Backward vs forward

| Method | Iteration `i` evaluates ... | "Important" features = ... |
|:-------|:-----------------------------|:---------------------------|
| `"backward"` | every subset of size `n - 1` (removing one feature at a time) | features whose **lower** importance value indicates earlier removal |
| `"forward"`  | every subset of size `i` (adding one feature at a time) | features whose **higher** importance value indicates earlier inclusion |

For both methods, the algorithm:

1. Builds all candidate feature subsets of the current size.
2. Fits a fresh `ClassificationModel` on each subset's data, cross-validates
   with `cv` folds, and records the mean score.
3. Selects the subsets within `best_score * (1 - uncertainty/100)`.
4. Updates each feature's "importance" counter.
5. Drops (backward) or accepts (forward) the selected feature(s) and
   loops to the next iteration.

## Return values

```python
importances, scores, best_importance, count_models = model.stepwise_regression(...)
```

- **`importances: dict[feature_name, int]`** &mdash; an integer rank.
  - In `backward` mode: lower means the feature was eliminated later
    (i.e. more important).
  - In `forward` mode: higher means the feature was selected earlier
    (i.e. more important).
- **`scores: dict[importance_rank, float]`** &mdash; the cross-validated
  score reached when each importance level was first observed. Useful for
  plotting accuracy as a function of "top-k features".
- **`best_importance: int`** &mdash; the importance rank that corresponds
  to the *peak* score across all iterations.
- **`count_models: int`** &mdash; total number of `ClassificationModel`
  fits performed (proxy for compute cost).

## Stopping criteria

The algorithm stops when any of these become true:

- Backward: only 1 feature remains.
- Forward: the selection from iteration `i` is identical to the selection
  from iteration `i - 1` (a fixed point).
- The current iteration's `best_score < threshold` (default `0.6`).
- `stop_if_worse=True` and the current iteration's score dropped more
  than `uncertainty` percent below the previous one.

## Pre-conditions / errors

| Condition | Exception |
|:----------|:----------|
| `points` empty | `Exception` |
| `len(features) < 2` | `ValueError` |
| `labels` empty | `Exception` |
| `len(points) != len(labels)` | `ValueError` |
| `method` not in `{"backward", "forward"}` | `ValueError` |
| `cv < 2` or `cv > len(points)` | `ValueError` |
| `retrain < 0` | `ValueError` |
| `threshold < 0.0 or > 1.0` | `ValueError` |
| `uncertainty < 0.0 or > 100.0` | `ValueError` |

## Compute cost

The number of models trained per iteration is `C(n, k)` where `n` is the
remaining feature count and `k` is the candidate subset size. For backward
elimination from `n=20` features, the first iteration trains 20 models;
forward selection from 1 to 2 features trains `C(20, 2) = 190` models on
iteration 2. This explodes quickly &mdash; use `n_jobs > 1` if the dataset
is small enough to benefit from parallelism.

## Worked example

```python
from sklearn.datasets import load_iris
from hdlib.model import ClassificationModel

iris = load_iris()
points = iris.data.tolist()
labels = iris.target.tolist()
features = iris.feature_names

model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
model.fit(points, labels=labels, seed=0)

importances, scores, best_importance, count_models = model.stepwise_regression(
    points=points,
    features=features,
    labels=labels,
    method="backward",
    cv=5,
    metric="accuracy",
    threshold=0.6,
    uncertainty=5.0,
    n_jobs=1,
)

print(f"trained {count_models} sub-models")
print(f"best importance (lower = better, backward): {best_importance}")
for f, imp in sorted(importances.items(), key=lambda kv: kv[1]):
    print(f"  {f}: {imp}")
```

### Reading the output

For backward elimination on Iris, you would typically see something like:

```
sepal length (cm): 6
sepal width  (cm): 8
petal length (cm): 2
petal width  (cm): 2
```

meaning petal-length and petal-width were retained longest (most
important), sepal-width was eliminated first.

## Pre-fitting requirement

The `stepwise_regression` method does **not** itself require `fit` to have
been called &mdash; it builds fresh `ClassificationModel`s internally via
`_init_fit_predict`. However, **inherited hyperparameters** (`self.size`,
`self.levels`, `self.vtype`) come from the receiver, so call the
constructor with the desired hyperparameters first.

If you want to find good hyperparameters *and* a good subset of features
in one shot, run `auto_tune` first (see `hdlib-hyperparameter-tuning`),
construct a fresh `ClassificationModel` with the tuned values, and then
call `stepwise_regression`.

## Common pitfalls

- **Score plateaus near the random baseline.** Either `threshold` was set
  too low (`stepwise_regression` keeps going), or your features are
  uninformative. Inspect `scores` to see the trajectory.
- **`uncertainty` too large.** Many subsets tie within the tolerance,
  so almost every feature ends up "important". Lower it to `1.0 - 2.0`.
- **Memory blowup with large datasets and `n_jobs > 1`.** Each fold spawns
  a process that pickles a fresh `ClassificationModel`. For
  `len(points) * size * levels` above a few hundred million ints,
  restrict `n_jobs`.
- **Forward selection on >>20 features.** The combinatorial explosion of
  `C(n, k)` makes forward selection impractical past 30-40 features.
  Prefer backward elimination for high-dimensional data.

## See also

- `hdlib-classification` &mdash; the underlying classifier.
- `hdlib-hyperparameter-tuning` &mdash; sweep `size` and `levels` before
  selecting features.
- `hdlib-encoding-data` &mdash; how features are encoded before selection.
