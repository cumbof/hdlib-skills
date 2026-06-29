---
name: hdlib-distance
description: Use when computing similarity / distance between hyperdimensional vectors with hdlib, choosing between cosine, Euclidean, and Hamming, picking a threshold for similarity search, or interpreting the distance returned by Vector.dist / Space.find / Space.find_all / GraphModel.edge_exists. Covers the trade-offs of each metric for bipolar vs binary vectors and the conventions hdlib uses.
---

# Distance metrics in hdlib

`hdlib` ships three distance implementations, all exposed through
`Vector.dist(other, method=...)` and consumed by every higher-level
similarity API (`Space.find`, `Space.find_all`, `GraphModel.edge_exists`,
`ClassificationModel.predict`, etc.).

| `method` value | Definition (NumPy)                                              | Range          | Notes                                                                                          |
|:---------------|:----------------------------------------------------------------|:---------------|:-----------------------------------------------------------------------------------------------|
| `"cosine"`     | `1 - dot(v1, v2) / (||v1|| * ||v2||)`                            | `[0, 2]`       | The default everywhere in hdlib. Two identical vectors give 0; orthogonal give 1; antipodal 2. |
| `"hamming"`    | `np.count_nonzero(v1 != v2)`                                     | `{0, ..., size}` | Integer count of mismatching components.                                                       |
| `"euclidean"`  | `np.linalg.norm(v1 - v2)`                                        | `[0, infinity)`     | Raw L2 norm.                                                                                   |

## When to use this skill

- The user calls `v1.dist(v2, method=...)`, `space.find(...)`, or
  `space.find_all(...)`.
- The user is picking a threshold for similarity search.
- The user is debugging why the "closest" vector returned is not what they
  expected (often a metric mismatch).
- The user is wondering why distances above `1.0` show up (cosine is
  `[0, 2]`, not `[0, 1]`).

## How distances are exposed across the library

```python
v1.dist(v2)                                  # method="cosine" by default
v1.dist(v2, method="hamming")

space.find(v)                                # (closest_name, cosine_dist)
space.find_all(v, threshold=0.5)             # only neighbours within cosine <= 0.5
space.find_all(v, method="hamming", threshold=200)

# Inside ClassificationModel.predict and stepwise_regression:
predict(test_indices, distance_method="cosine", retrain=0)
```

## Picking a metric

### Bipolar vectors (`vtype="bipolar"`)

- **Cosine (default).** Recommended for HDC. For unit-length random
  bipolar vectors, near-orthogonal pairs sit around `cosine ~= 1.0` and
  identical pairs at `0.0`. The default for every model.
- **Hamming.** Fine, but returns an integer count that scales with `size`,
  making thresholds hard to reason about. Useful only for diagnostics or
  when you've intentionally stored very few-bit vectors.
- **Euclidean.** Almost never the right choice for bipolar vectors &mdash;
  random pairs sit at `~2 * sqrt(size)` and identical pairs at `0`. It is
  proportional to `sqrt(hamming)` but offers no clear advantage.

### Binary vectors (`vtype="binary"`)

- **Hamming.** The natural choice. For random binary vectors with
  `D=10000`, expect about `5000` differing bits.
- **Cosine.** Works, but interpretation is less intuitive than Hamming.
  Identical pairs return `0`; orthogonal pairs return `1`; antipodal
  pairs (all-zero versus all-one) return `2`.
- **Euclidean.** Same caveats as for bipolar.

## Cosine distance values you should expect

```python
import numpy as np
from hdlib.vector import Vector

D = 10000
v1 = Vector(seed=1, size=D)
v2 = Vector(seed=2, size=D)

# Identical
print(v1.dist(v1))          # 0.0

# Independent random pair (near orthogonal)
print(v1.dist(v2))          # ~1.0 (typically 0.99 .. 1.01)

# Negated
v3 = Vector(vector=-v1.vector.copy(), size=D, vtype="bipolar")
print(v1.dist(v3))          # 2.0 (antipodal)
```

## Choosing a similarity threshold

`Space.find_all` and `GraphModel.edge_exists` both accept a `threshold`:
vectors within `dist <= threshold` are kept.

Empirical thresholds (bipolar, `size=10000`):

- `cosine <= 0.4` &mdash; "very similar". Survives bundling of 2 - 3 components.
- `cosine <= 0.6` &mdash; "related". Survives bundling of 5 - 10 components.
- `cosine <= 0.9` &mdash; "marginally similar". Use as a loose filter.
- `cosine >= 1.0` &mdash; "orthogonal-or-worse". Filter out.

For binary vectors, divide by `size`:

- `hamming <= 0.4 * size` &mdash; very similar.
- `hamming <= 0.5 * size` &mdash; chance baseline (treat as unrelated).

## `Space.find` returns *minimum* distance

`Space.find` and `find_all` iterate over every vector in the space and
return the one with the smallest distance, breaking ties by insertion order
(since the underlying loop keeps the first vector that strictly improved
the running minimum).

If the space is **empty**, `find_all` returns `({}, None)` and `find`
returns `(None, np.inf)`.

If `threshold` is set and no vector falls under it, `find_all` returns
`({}, None)` and `find` returns `(None, np.inf)`.

## Distance vs similarity

hdlib reports **distances**, not similarities. Convert by:

```python
cosine_similarity = 1.0 - cosine_distance
```

The `RegressionModel._calculate_similarity` does exactly this internally.

## Common pitfalls

- **Treating cosine as `[0, 1]`.** It is `[0, 2]`. Antipodal vectors yield
  `2.0`. If you cap your threshold at `1.0` you ignore the more
  informative half of the range.
- **Mixing vtypes** &mdash; raises `Exception("Vectors must be of the same type")`.
  Both operands must have the same `vtype`.
- **Mismatched `size`** &mdash; raises `Exception("Vectors must have the same size")`.
- **Comparing un-normalised bundles.** A bundled vector with components in
  `{-3, -1, 1, 3}` still has a well-defined cosine distance, but its
  magnitude makes Euclidean distance non-comparable across operations.
  Normalise before comparing if you mixed bundling and individual vectors.
- **Hamming distance is an integer count** &mdash; for `D=10000` random
  bipolar vectors expect ~5000. Divide by `size` if you need a ratio.
- **Cosine of a zero vector is NaN.** A binary vector that has been
  normalised down to all zeros has `||v|| = 0`. The dist call wraps NumPy
  in an `errstate("ignore")` context inside `ClassificationModel.error_rate`
  to silence the warning, but downstream code may treat the NaN as a
  failure.

## Worked example &mdash; comparing the three metrics on the same pair

```python
import numpy as np
from hdlib.vector import Vector

v1 = Vector(seed=11)
v2 = Vector(seed=42)

for method in ("cosine", "euclidean", "hamming"):
    print(method, v1.dist(v2, method=method))
```

Typical output for two random bipolar vectors of size 10,000:

```
cosine      0.9967
euclidean   141.13
hamming     4977
```

`cosine ~= 1` and `hamming ~= 5000` both communicate "essentially
orthogonal random vectors" &mdash; pick whichever is more readable for the task.

## See also

- `hdlib-vectors` for how `dist` is wired up to the `Vector` class.
- `hdlib-space` for `Space.find` and `Space.find_all` &mdash; they delegate
  here for the actual comparison.
- `hdlib-classification`, `hdlib-graph` and `hdlib-clustering` for models
  that take `distance_method` as a parameter and apply it identically to
  every vector pair internally.
