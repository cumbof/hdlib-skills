---
name: hdlib-encoding-data
description: Use when deciding how to turn real-world data (tabular numerical, categorical, sequences, k-mers, SMILES, graphs, time series) into hyperdimensional vectors that hdlib's models can consume. Covers the level-vector pattern (ClassificationModel.fit), the cos*sin RegressionEncoder, the projection-matrix encoder (ClusteringModel), and the role-filler / sequence / graph patterns built from bind, bundle, permute. Load whenever the user has data in non-vector form and needs to feed it to hdlib.
---

# Encoding data into hyperdimensional vectors

`hdlib` does **not** ship a data loader. It expects either `Vector`
objects or, where models accept raw data, plain Python `list[list[float]]`
or `numpy.ndarray`. This skill catalogues the standard patterns to go
from raw data to `Vector` / numpy form so the models can consume it.

## When to use this skill

- The user has tabular data with numerical and / or categorical features.
- The user has DNA / RNA / protein sequences, SMILES strings, or any
  token stream.
- The user wants to encode a graph, knowledge graph, or time series.
- The user is reaching for `Vector(vector=...)` with no idea what to put
  inside.

## Decision tree (pick one encoder)

1. **Numerical features, supervised classification** -> use
   `ClassificationModel.fit(points, labels)`. The model has a built-in
   level-vector encoder (see Pattern A below).
2. **Numerical features, unsupervised clustering** -> use
   `ClusteringModel`. It applies a random projection + sign (Pattern C).
3. **Numerical features, regression target** -> use `RegressionEncoder`
   (Pattern B).
4. **Categorical / symbolic features** -> hand-roll role-filler
   bindings (Pattern D).
5. **Sequences** (DNA, SMILES, words) -> bundle of permuted token
   vectors (Pattern E).
6. **Graphs** -> use `GraphModel` (Pattern F) or hand-roll
   adjacency bundles.

## Pattern A &mdash; Level-vector encoding (built into `ClassificationModel`)

For each feature value:

1. Map the value to one of `levels` equal-width bins between the global
   `min_value` and `max_value` of the dataset.
2. Look up the corresponding `level_<i>` vector.
3. Permute it by the feature index (to encode position).
4. Bundle all permuted level vectors across features.

You don't have to write this code: just call `model.fit(points, labels)`.
For reference:

```python
def _encode_point(self, point):
    sum_vector = None
    for i, value in enumerate(point):
        level = bin_index(value, self.min_value, self.max_value, self.levels)
        level_vec = self.space.get(names=[f"level_{level}"])[0]
        rolled = permute(level_vec, rotate_by=i)
        sum_vector = rolled if sum_vector is None else bundle(sum_vector, rolled)
    return sum_vector
```

Pre-conditions to feed `ClassificationModel.fit`:

- Each row is a `list[float]`.
- All features are roughly on the same scale &mdash; the model uses a single
  global `(min_value, max_value)`. Normalise or standardise first.
- No NaNs.

See `hdlib-classification` for the full API.

## Pattern B &mdash; `RegressionEncoder` (cos*sin)

```python
from hdlib.model.regression import RegressionEncoder
encoder = RegressionEncoder(D=10000, n_features=8)
hv = encoder.encode(feature_row)        # shape (D,), real-valued
```

The mapping is `h_i = cos(F . B_i + b_i) * sin(F . B_i)`. The encoder is
similarity-preserving and works best on **`[0, 1]`-scaled** features.

This is internally driven by `RegressionModel`; you can also use it
standalone to feed an HD regression head you trained elsewhere.

## Pattern C &mdash; Random projection encoder (`ClusteringModel`)

`ClusteringModel.fit` projects each row through a fixed Gaussian random
matrix and then `np.sign`s the result to obtain a bipolar HD vector.
You don't need to write this yourself, but if you want HDC embeddings
for downstream use:

```python
import numpy as np
from hdlib.vector import Vector

D = 10000
projection = np.random.default_rng(0).standard_normal((n_features, D))
def encode(row):
    proj = np.dot(row, projection)
    bipolar = np.sign(proj)
    bipolar[bipolar == 0] = 1
    return Vector(vtype="bipolar", vector=bipolar.astype(int))
```

This is simpler than the level-vector encoder and natively supports any
numerical range, but it does **not** preserve magnitude ordering as
cleanly as Pattern A.

## Pattern D &mdash; Role-filler bindings for categorical data

For each categorical feature `f` with value `v`, build:

```
record = bundle( bind(ROLE_f, FILLER_v)  for each feature f )
```

- Pre-generate one `Vector` per role (one per feature name).
- Pre-generate one `Vector` per possible filler value (per categorical
  level).
- Bundle the bindings into a single record vector.

```python
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle

D = 10000
space = Space(size=D, vtype="bipolar")

roles    = ["COLOR", "SHAPE", "SIZE"]
fillers  = {"COLOR": ["red", "green", "blue"],
            "SHAPE": ["circle", "square", "triangle"],
            "SIZE":  ["small", "medium", "large"]}

for r in roles:
    space.insert(Vector(name=r, size=D, vtype="bipolar"))
    for f in fillers[r]:
        space.insert(Vector(name=f"{r}:{f}", size=D, vtype="bipolar"))

def encode(sample):
    record = None
    for role, filler in sample.items():
        b = bind(space.get(names=[role])[0],
                 space.get(names=[f"{role}:{filler}"])[0])
        record = b if record is None else bundle(record, b)
    record.normalize()
    return record

sample = {"COLOR": "red", "SHAPE": "circle", "SIZE": "large"}
hv = encode(sample)
```

This is the pattern behind the "Dollar of Mexico" reasoning skill
(`hdlib-analogical-reasoning`). It generalises to any
attribute-value structure.

## Pattern E &mdash; Sequence encoding

For token streams (words, k-mers, characters), encode position with
`permute` and bundle:

```python
def encode_sequence(tokens, token_to_vec):
    seq = None
    for i, tok in enumerate(tokens):
        rolled = permute(token_to_vec[tok], rotate_by=i)
        seq = rolled if seq is None else bundle(seq, rolled)
    seq.normalize()
    return seq
```

Pre-generate one `Vector` per token type once and reuse it across all
sequences (otherwise the encoding is non-comparable). For long sequences
(>200 tokens), bundle capacity erodes; consider sliding windows or
hierarchical encoding.

For DNA k-mers, use the token vocabulary `[A, C, G, T]^k`. For SMILES,
use the atom-wise or k-mer tokeniser from
`examples/tox21/tox21.py` in the hdlib repo.

## Pattern F &mdash; Graph encoding via `GraphModel`

```python
from hdlib.model import GraphModel
edges = {(src, dst, weight), ...}     # one tuple per edge
graph = GraphModel(size=10000, directed=False, seed=0)
graph.fit(edges)
```

`GraphModel.fit` builds one vector per unique node, one per unique
weight, then bundles each node's `(weight * neighbour)` pairs to form
its memory. Finally it bundles every `node * permute(memory, 1)` (or
`node * memory` for undirected) into a single graph vector.

You can then call `graph.edge_exists(...)` or `graph.predict(...)`.
See `hdlib-graph`.

If your graph nodes carry **features**, encode each node's features
first (e.g. Pattern A or D) into a single `Vector` per node, then plug
those into a custom graph encoder rather than letting `GraphModel`
generate random per-node vectors. The library doesn't provide a built-in
feature-aware graph encoder, but the building blocks are exposed.

## Pattern G &mdash; Time-series encoding

Two options:

1. **Bin each timestep into a level vector** (Pattern A) and bundle
   permuted timestep vectors (Pattern E). This is the standard
   "univariate time-series HDC" pattern.
2. **Apply Pattern C** (random projection) to each timestep's feature
   vector and then bundle with permutation.

```python
def encode_timeseries(ts_rows, level_vec_lookup):
    seq = None
    for t, row in enumerate(ts_rows):
        rolled = permute(level_vec_lookup(row), rotate_by=t)
        seq = rolled if seq is None else bundle(seq, rolled)
    seq.normalize()
    return seq
```

For long series, downsample first &mdash; bundle capacity caps useful
length at a few hundred timesteps.

## Common pitfalls

- **Mixing scales.** A dataset with features in `[0, 1]` and others in
  `[0, 1e6]` collapses the small-range features into 1 - 2 level bins
  when fed to `ClassificationModel.fit`. Standardise or min-max-scale
  first.
- **Re-randomising token vectors across samples.** The token vocabulary
  must be **shared** across every encoded sequence; otherwise distances
  between sequences are meaningless. Build the vocabulary once,
  preferably in a `Space`, and reuse it.
- **Skipping `normalize()` after bundling many vectors.** The composite
  may overflow into `{-N, ..., N}`. Most hdlib distance code handles it,
  but downstream code that asserts `vtype="bipolar"` strictly does not.
- **Categorical encoders for high-cardinality features.** Allocating
  10k random vectors for a 10k-cardinality feature works but explodes
  memory. Hash-bucket categorical values into a fixed-size vocabulary
  first (e.g. 1024 buckets via `hash(value) % 1024`).
- **Forgetting that permute is by integer positions.** Using float
  positions or negative wraparound can produce unexpected behaviour.
  `permute(v, rotate_by=k)` calls `numpy.roll(v.vector, k, axis=0)`.

## See also

- `hdlib-classification` for the level-vector encoder reference.
- `hdlib-regression` for the cos*sin encoder reference.
- `hdlib-clustering` for the random projection encoder reference.
- `hdlib-graph` for graph encoding.
- `hdlib-analogical-reasoning` for the role-filler pattern.
- `hdlib-arithmetic` for the `bundle` / `bind` / `permute` mechanics.
