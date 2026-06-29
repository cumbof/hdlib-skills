---
name: hdlib-overview
description: Master skill for the hdlib Python library (Hyperdimensional Computing / Vector-Symbolic Architectures). Load this first whenever the user mentions hdlib, hyperdimensional computing, HDC, or Vector-Symbolic Architectures and you do not yet know which specific feature is needed. Provides the architecture map and tells you which other hdlib-* skill to load next.
---

# hdlib at a glance

`hdlib` is a Python 3 library for building **Vector-Symbolic Architectures
(VSA)**, also called **Hyperdimensional Computing (HDC)**: high-dimensional
(thousands of components) vectors that represent symbols and concepts, and
algebraic operators that combine them. Authored by Fabio Cumbo et al.

- PyPI / Conda package name: `hdlib`
- Repository: <https://github.com/cumbof/hdlib>
- Current target version for these skills: **2.1.0**
- License: MIT
- Distributed alongside the Joss paper [`Cumbo et al. 2023`](https://doi.org/10.21105/joss.05704)
  and an extended arXiv preprint [`Cumbo et al. 2026`](https://arxiv.org/abs/2601.02509).

## Public API map

```
hdlib/
├── __init__.py             # exposes __version__ (currently "2.1.0")
├── vector.py               # Vector
├── space.py                # Space (re-exports Vector)
├── arithmetic/
│   ├── __init__.py         # bind, bundle, subtraction, permute  (classical)
│   └── quantum.py          # encode, bind, bundle, permute, compress_circuit,
│                           # statevector_to_bipolar, run_compute_uncompute_test,
│                           # get_circuit_metrics, superposition_bundle,
│                           # entangled_bind, grover_search,
│                           # quantum_majority_bundle, quantum_contextual_bind
└── model/
    ├── __init__.py         # re-exports the model classes below
    ├── classification.py   # ClassificationModel, QuantumClassificationModel
    ├── clustering.py       # ClusteringModel
    ├── graph.py            # GraphModel
    └── regression.py       # RegressionEncoder, RegressionModel
```

### Canonical imports

```python
# Core
from hdlib import __version__
from hdlib.vector import Vector
from hdlib.space import Space

# Classical MAP arithmetic
from hdlib.arithmetic import bind, bundle, subtraction, permute

# Models (re-exported via hdlib.model)
from hdlib.model import (
    ClassificationModel,
    QuantumClassificationModel,
    ClusteringModel,
    GraphModel,
    RegressionEncoder,
    RegressionModel,
)

# Quantum arithmetic
from hdlib.arithmetic.quantum import (
    encode, bind as q_bind, bundle as q_bundle, permute as q_permute,
    compress_circuit, statevector_to_bipolar, run_compute_uncompute_test,
    get_circuit_metrics, superposition_bundle, entangled_bind, grover_search,
    quantum_majority_bundle, quantum_contextual_bind,
)
```

## When to load which skill

Match the user's intent to the most specific skill below. Load `hdlib-overview`
only when there is no specific skill that fits.

| User intent | Load |
|:------------|:-----|
| Install hdlib, deal with dependencies, package issues | `hdlib-installation` |
| Create vectors, combine them, compute distance | `hdlib-vectors` (and `hdlib-arithmetic`) |
| Maintain a dictionary-like collection of vectors with tags / links / search | `hdlib-space` |
| Apply the MAP operators `bundle` / `bind` / `subtraction` / `permute` | `hdlib-arithmetic` |
| Compare two vectors with cosine / Euclidean / Hamming | `hdlib-distance` |
| Train a classifier on tabular numerical data | `hdlib-classification` |
| Rank features by importance | `hdlib-feature-selection` |
| Tune vector dimensionality and number of level vectors | `hdlib-hyperparameter-tuning` |
| Cluster unlabelled numerical data | `hdlib-clustering` |
| Predict a continuous target | `hdlib-regression` |
| Encode a graph / pangenome / network and query edges | `hdlib-graph` |
| Use Qiskit-backed quantum arithmetic | `hdlib-quantum-arithmetic` |
| Train the quantum classification model on simulator or IBM hardware | `hdlib-quantum-classification` |
| Use compute-uncompute test, superposition bundle, Grover search, entangled bind | `hdlib-quantum-advanced` |
| Implement the "Dollar of Mexico" / role-filler analogy pattern | `hdlib-analogical-reasoning` |
| Decide how to encode tabular / sequence / graph data | `hdlib-encoding-data` |
| Debug unexpected outputs or errors | `hdlib-pitfalls` |
| Make results deterministic across runs | `hdlib-reproducibility` |

## Core concepts (just enough to disambiguate other skills)

- **Hyperdimensional vector**: a (very) long 1D vector, typically of size
  `D = 10_000`. Components are either `{0, 1}` (binary / Boolean) or `{-1, +1}`
  (bipolar). hdlib refers to this choice as **`vtype`** ("binary" or "bipolar"
  &mdash; bipolar is the default everywhere).
- **Space**: a hashmap of `name -> Vector`. All vectors in the same space must
  share the same `size` and `vtype`. The space also indexes vectors by
  user-supplied tags and supports directed links.
- **MAP arithmetic** (Multiply / Add / Permute):
  - `bundle(a, b)` &mdash; **add** &mdash; produces a vector *similar* to both inputs.
  - `bind(a, b)` &mdash; **multiply** &mdash; produces a vector *dissimilar* to both,
    invertible (unbind = bind again with one of the inputs).
  - `permute(v, k)` &mdash; **rotate** &mdash; encodes order / position.
  - `subtraction(a, b)` &mdash; element-wise difference (bipolar only).
- **Level vectors** (`ClassificationModel.fit`): a set of `levels`
  pre-generated vectors quantising a numerical range; each numeric feature
  value maps to one of them.
- **Quantum HDC**: hdlib also represents hypervectors as **phase oracles**
  in a Qiskit `QuantumCircuit`. Bipolar values `+1`/`-1` become phases
  `0`/`pi`. The library reproduces every MAP operator on circuits and adds
  algorithms (compute-uncompute similarity, Grover search, ...).

## Minimum-viable hdlib snippet

```python
from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic import bind, bundle

space = Space(size=10000, vtype="bipolar")
space.bulk_insert(names=["alice", "bob", "carol"])
alice, bob, carol = (space.get(names=[n])[0] for n in ("alice", "bob", "carol"))

# Build a composite "group" by bundling the three vectors
group = bundle(bundle(alice, bob), carol)
group.name = "group"
space.insert(group)

# Find the closest existing vector to the group
best, dist = space.find(group)
print(best, dist)  # one of alice / bob / carol with a small cosine distance
```

## Common patterns flagged across multiple skills

- **Always match `size` and `vtype` between any two vectors** you combine or
  insert into the same space &mdash; mismatches raise plain `Exception`s.
  See `hdlib-pitfalls`.
- After bundling/binding, the resulting vector may contain values outside
  `{-1, +1}` or `{0, 1}`; call `vector.normalize()` to project it back to
  the canonical range. See `hdlib-arithmetic`.
- For deterministic vectors, pass an integer `seed` to `Vector(...)`. For
  deterministic *models*, also pass `seed=...` to `ClassificationModel.fit`,
  `GraphModel(...)`, `ClusteringModel(...)`. See `hdlib-reproducibility`.
- The quantum arithmetic accepts only **bipolar** vectors with sizes that are
  a power of 2. See `hdlib-quantum-arithmetic`.

## When *not* to use hdlib

- You only need a single linear classifier on a small, dense dataset &mdash; a
  conventional library (scikit-learn) will be faster and simpler.
- You need GPU-accelerated training out of the box (hdlib stays on CPU /
  Qiskit; the `AerSimulator` can use a GPU if available, but the classical
  paths do not).
- You want continuous regression with a tight error budget &mdash; the RegHD
  model in `hdlib.model.regression` works but is approximate.

## See also

- `hdlib-installation` &mdash; getting the library running locally.
- `hdlib-vectors`, `hdlib-space`, `hdlib-arithmetic`, `hdlib-distance` &mdash;
  the foundations every other skill builds on.
- Model and quantum skills listed in the [Skills index](../README.md).
