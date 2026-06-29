# Smoke tests for `hdlib-skills`

This directory contains a single smoke-test script
[`smoke_test.py`](./smoke_test.py) that exercises one canonical code
block from every skill in `../skills/`. It verifies that the public
API examples in each skill actually run against the installed
`hdlib` version.

## Running

```bash
# 1. Set up a virtual environment with hdlib
python -m venv .venv
source .venv/bin/activate
pip install 'hdlib==2.1.0'        # or the version under test

# 2. Run the script
python test/smoke_test.py
```

Expected output (last line on success):

```
All 18 skill smoke tests passed.
```

## What it covers

| Test function | Validates skill |
|:--------------|:----------------|
| `test_overview` | `hdlib-overview` |
| `test_vectors` | `hdlib-vectors` |
| `test_space` | `hdlib-space` |
| `test_arithmetic` | `hdlib-arithmetic` |
| `test_distance` | `hdlib-distance` |
| `test_classification` | `hdlib-classification` |
| `test_feature_selection` | `hdlib-feature-selection` |
| `test_hyperparameter_tuning` | `hdlib-hyperparameter-tuning` |
| `test_clustering` | `hdlib-clustering` |
| `test_regression` | `hdlib-regression` |
| `test_graph` | `hdlib-graph` |
| `test_quantum_arithmetic` | `hdlib-quantum-arithmetic` |
| `test_quantum_classification` | `hdlib-quantum-classification` |
| `test_quantum_advanced` | `hdlib-quantum-advanced` |
| `test_analogical_reasoning` | `hdlib-analogical-reasoning` |
| `test_encoding_data` | `hdlib-encoding-data` |
| `test_reproducibility` | `hdlib-reproducibility` |
| `test_pitfalls` | `hdlib-pitfalls` |

Each test prints a `PASS <skill-name>` line on success, or `FAIL`
with the exception. Failures are summarised at the end.

## Real hdlib gotchas surfaced by this test

Running this script against `hdlib==2.1.0` uncovered (and the skills
now document) the following library-level issues:

- `Space.bulk_insert(names=..., tags=...)` misaligns names and tags due
  to a `set(names)` reorder. The skills route around it via per-vector
  `insert` or post-hoc `add_tag`.
- `hdlib.arithmetic.quantum.bundle(..., method="classical")` does **not**
  round-trip through `statevector_to_bipolar` as a clean majority vote;
  it preserves magnitude via RMS-scaled phases. The default
  `method="average"` does round-trip.
- `QuantumClassificationModel.__init__` constructs
  `AerSimulator(device="GPU")` and only catches construction errors. On
  systems without CUDA the construction succeeds, but `predict` fails at
  run time. Force the simulator to CPU after construction:
  `model.backend = AerSimulator()`.
- `GraphModel(seed=...)` only seeds the auto-threshold sampling; per-node
  `Vector(...)` calls inside `_add_edge` are unseeded. Pre-build the
  node vectors with `Vector(seed=...)` and inject `memory` and
  `weights` attributes manually for full determinism.

## Adding a new test

When you add a new skill, drop a `test_<skill_short_name>` function into
`smoke_test.py` and append it to the `tests` list in the `__main__`
block. Keep test data small so the whole script stays under a few
minutes (currently ~30 - 60 seconds on a modern laptop).
