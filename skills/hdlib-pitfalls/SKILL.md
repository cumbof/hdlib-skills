---
name: hdlib-pitfalls
description: Use when debugging unexpected errors, surprising results, or silent misbehaviour in hdlib code - mismatched vector sizes or vtypes, missing normalization after bundling, refusal to overwrite pickle files, version skew warnings from loaded pickles, multiprocessing issues with auto_tune / cross_val_predict, NaN cosine distances, set-ordering bugs, and quantum power-of-2 requirements. Load whenever the user gets a TypeError / ValueError / Exception from anywhere in hdlib or asks "why is this returning wrong results?"
---

# hdlib troubleshooting and common pitfalls

A consolidated catalogue of the failure modes that actually bite users
of `hdlib`. Each entry includes the symptom, the root cause inside the
hdlib source, and the fix.

## When to use this skill

- The user hits an exception traceback from `hdlib/...`.
- The model's predictions / distances look wrong but no exception was
  raised.
- The user asks "why is this not reproducing across runs?"
- The user is fighting multiprocessing, pickle round-trips, or version
  mismatches.

## 1. `Exception: Vectors must have the same size`

**Where:** `hdlib/arithmetic/__init__.py` (every operator), `hdlib/vector.py:Vector.dist`, `hdlib/space.py:Space.find_all`.

**Cause:** You combined two `Vector` objects with different `.size`, or
inserted a `Vector` of size X into a `Space(size=Y)`.

**Fix:** Always inherit the size from the `Space` you're working in.
Prefer `Space.bulk_insert` (which uses `space.size` automatically) to
hand-constructing `Vector(size=...)`.

## 2. `Exception: Vector types are not compatible`

**Where:** Arithmetic operators and distance.

**Cause:** Mixing `vtype="binary"` and `vtype="bipolar"`.

**Fix:** Pick one vtype per space. The default is `bipolar`. Only switch
to `binary` if you have a concrete reason (e.g. matching an external
binary encoding).

## 3. `Exception: Subtraction is not available for binary vectors`

**Where:** `hdlib/arithmetic/__init__.py:subtraction`.

**Cause:** Tried to subtract two binary Vectors. The library forbids
this because there is no obvious binary-arithmetic interpretation.

**Fix:** Use `bind` (XOR) if you want to express "differing positions";
or convert to bipolar first.

## 4. `Exception: The output file already exists!`

**Where:** `Vector.dump`, `Space.dump`.

**Cause:** Both classes refuse to overwrite an existing file as a safety
measure.

**Fix:** Either choose a fresh path or `os.remove(path)` first:

```python
import os
if os.path.exists(path):
    os.remove(path)
vector.dump(to_file=path)
```

## 5. `Warning: the specified Vector/Space has been created with a different version of hdlib`

**Where:** Stdout, on `Vector(from_file=...)` and `Space(from_file=...)`.

**Cause:** The pickle was produced by a different hdlib version. The
load still proceeds, but the API surface may have changed under your
feet (the pickle restores the `__dict__` verbatim).

**Fix:** Regenerate the pickle on the running version, or pin both
producer and consumer to the same `hdlib==X.Y.Z`.

## 6. Components outside `{-1, +1}` after `bundle` / `subtraction`

**Where:** `bundle` returns the raw element-wise sum (bipolar) without
normalising; `subtraction` returns the raw element-wise difference.

**Symptom:** A "bipolar" vector that contains 0, +2, -2, or larger.

**Cause:** This is *by design*, but downstream code that strictly
expects `+/-1` (your own assertions, or e.g. quantum encoders that
demand bipolar input) will break.

**Fix:** Call `vector.normalize()` after composition:

```python
v_total = bundle(bundle(a, b), c)
v_total.normalize()      # back to {-1, +1}
```

## 7. NaN cosine distance

**Where:** `Vector.dist(other, method="cosine")` when one of the operands
is the all-zero vector.

**Cause:** `||v|| == 0` triggers a divide-by-zero in
`np.dot(v, w) / (||v|| * ||w||)`.

**Fix:** Skip cosine on zero vectors. The classification model wraps the
call in `with np.errstate(invalid="ignore", divide="ignore"):` to
silence the runtime warning but still produces `NaN`. Treat `NaN` as
"infinite distance" if you must compute it.

## 8. `space.find` returns the wrong vector

**Possible causes:**

- The query vector was never normalised (see Pitfall 6); cosine still
  works but Euclidean/Hamming numbers are misleading.
- The query vector has tags set directly via `query.tags.add(...)`
  &mdash; the space's tag index is stale; `find` won't be wrong but
  `space.get(tags=...)` will.
- You inserted multiple vectors with similar content (e.g. many bundles
  of overlapping component sets) and the ordering of similarities is
  tighter than you expected. Inspect `find_all` distances.
- Insertion-order tie-breaking: when two vectors have *identical*
  distance to the query, `find_all` keeps the first one it saw to be
  strictly smaller than the running minimum &mdash; effectively first-in
  wins on ties.

## 9. `multiprocessing.Pool` errors on macOS / Jupyter

**Where:** `ClassificationModel.cross_val_predict`,
`ClassificationModel.auto_tune`,
`ClassificationModel.stepwise_regression`,
`GraphModel.error_mitigation`.

**Symptom:** "fork is being used on a platform that..." or pickling
errors involving local functions / lambdas.

**Cause:** The hdlib code uses `multiprocessing.Pool` (which on macOS
defaults to "spawn"). The forked process needs to be able to import the
module containing your top-level code.

**Fix:** Run your script with the standard `if __name__ == "__main__":`
guard. In Jupyter, set `n_jobs=1` to avoid the issue.

## 10. `auto_tune` returned `(None, None, None)`

**Where:** `ClassificationModel.auto_tune`.

**Cause:** The candidate filter `if size > len(points) and levels > 1`
discarded every combination. The most common reason is
`min(size_range) <= len(points)`.

**Fix:** Ensure `size_range` starts above your sample count, e.g.
`range(max(1000, len(points) + 1), 10001, 1000)`.

## 11. Non-deterministic runs

**Where:** Anywhere randomness creeps in.

**Cause categories:**

- `Vector(...)` without `seed=...` uses fresh randomness each time.
- `ClassificationModel.fit` requires `seed=...` for reproducible level
  vectors.
- `GraphModel(seed=...)` controls only the auto-threshold sampling in
  `edge_exists`.
- `ClusteringModel(seed=...)` seeds NumPy's **global** RNG once at
  construction; subsequent fits drift.
- `RegressionEncoder` and `RegressionModel.cluster_models_b` use the
  global NumPy RNG &mdash; `np.random.seed(...)` immediately before
  constructing.

**Fix:** See `hdlib-reproducibility` for the seed map across the library.

## 12. `_encode_point` produces identical vectors for all rows

**Where:** Inside `ClassificationModel.fit` if you forgot to scale.

**Cause:** `_encode_point` computes the bin index by comparing each
feature value to `(self.min_value, self.max_value)` ranges. If your
features have wildly different scales (e.g. age in `[0, 100]` and income
in `[0, 1e6]`), the smaller-scale features collapse into one or two
bins and the encoded vectors lose information.

**Fix:** Min-max scale your features to `[0, 1]` (or standardise to
zero-mean unit-variance) **before** calling `fit`.

## 13. `QuantumClassificationModel` constructor raises `ValueError: vector dimensionality must be a power of 2`

**Cause:** `size` is checked via `(size > 0) and (size & (size - 1) == 0)`.

**Fix:** Use `size in {32, 64, 128, 256, 512, 1024}`. The classical
default `10000` is **not** a power of 2 and cannot be reused here.

## 14. `levels` features being collapsed by min/max global range

**Where:** `ClassificationModel.fit`.

**Cause:** The `min_value` and `max_value` are computed globally across
*all* features in *all* samples. A single outlier feature with extreme
range squashes everything else into level 0 or level `levels-1`.

**Fix:** Clip or winsorise outliers, or rescale features per-feature
before passing them.

## 15. `Vector` renamed after `Space.insert(...)`

**Symptom:** `name in space` returns `False`; `space.get(names=[v.name])`
returns `[]`.

**Cause:** `Space.insert` indexes by the name *at insertion time*.
Mutating `vector.name` afterwards does not update the index.

**Fix:** Always `space.remove(old_name) -> rename -> space.insert(...)`.

## 16. `Vector.tags` mutated directly &mdash; tag search misses the vector

**Symptom:** `space.get(tags=[tag])` returns `[]` even though
`vector.tags` contains the tag.

**Cause:** `vector.tags.add(tag)` doesn't update `space.tags[tag]`.

**Fix:** Use `space.add_tag(name, tag)` and `space.remove_tag(name, tag)`.

## 17. `Space(size=10)` silently accepts a tiny size

**Cause:** Pre-2.0 hdlib enforced `size >= 1000`; the current code
removed that check.

**Fix:** Use `size >= 1000` yourself for sensible HDC behaviour;
preferably `size >= 5000` for non-trivial models.

## 18. `set`-order surprises

**Where:** `ClassificationModel.classes`, `Space.tags[<tag>]`,
`Space.get(tags=...)`.

**Cause:** Many internal containers are Python `set`s; iteration order
is insertion order in CPython 3.7+ but should not be relied on.

**Fix:** When you need a deterministic order (e.g. for the columns of a
distance matrix), `sorted(model.classes)` or read from
`class_vectors[i].tags`.

## 19. Quantum encode pads silently

**Where:** `hdlib.arithmetic.quantum.encode`.

**Cause:** If `len(vec)` is not a power of 2, the function pads with
`+1` up to the next power of 2. The padded positions affect all
downstream similarity calculations.

**Fix:** Construct quantum-side bipolar vectors with sizes in
`{16, 32, ..., 1024}` only.

## 20. `chopin2`-style example fails on dependency import

**Symptom:** `ImportError: cannot import name 'tabulate'` or similar.

**Cause:** The `examples/` scripts import optional dependencies
(`tabulate`, `pandas`, `rdkit`, `tensorflow`) that hdlib does not pin
beyond what is in `requirements.txt`.

**Fix:** Install the example's specific dependencies; do not assume
that `pip install hdlib` brings them in.

## See also

- `hdlib-reproducibility` for the seed map.
- `hdlib-installation` for installation-level errors.
- Individual feature skills for feature-specific quirks.
