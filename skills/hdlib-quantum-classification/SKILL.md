---
name: hdlib-quantum-classification
description: Use when building, fitting, predicting with, or retraining the quantum hyperdimensional classifier hdlib.model.classification.QuantumClassificationModel. Covers the constructor (size as power of 2, levels, seed, shots, IBM channel/instance/backend/api_key, noise_model_from), the level vector generation, the compute-uncompute test that drives prediction, sample-wise sample encoder construction, per-class prototype circuits, the perceptron-style retrain method, and the differences between running on AerSimulator vs IBM Quantum hardware.
---

# Quantum classification with `QuantumClassificationModel`

`hdlib.model.classification.QuantumClassificationModel` is the
Qiskit-backed counterpart of `ClassificationModel`. It encodes each
training sample as a sequence of phase-oracle circuits, bundles those
circuits into a per-class **prototype circuit**, and answers queries with
the compute-uncompute test from `hdlib.arithmetic.quantum`. The model
runs on either the local `AerSimulator` (with optional noise model) or
real IBM Quantum hardware via `qiskit-ibm-runtime`.

Based on
[Cumbo et al. 2025 - Quantum Hyperdimensional Computing](https://doi.org/10.48550/arXiv.2511.12664).

## When to use this skill

- The user instantiates `QuantumClassificationModel(...)`.
- The user calls `.fit`, `.predict`, or `.retrain` on a quantum model.
- The user wants to submit jobs to an IBM Quantum backend.
- The user is comparing classical `ClassificationModel` with the quantum
  variant on the same dataset.

For the underlying quantum primitives (encode, bundle, bind, permute,
compute-uncompute) see `hdlib-quantum-arithmetic` and
`hdlib-quantum-advanced`.

## Import

```python
from hdlib.model import QuantumClassificationModel
# or
from hdlib.model.classification import QuantumClassificationModel
```

## Constructor

```python
QuantumClassificationModel(
    size: int = 64,                # MUST be a power of 2
    levels: int = 2,
    seed: int = 42,
    shots: int = 1024,
    channel: Optional[str] = None,           # e.g. "ibm_quantum_platform"
    instance: Optional[str] = None,
    backend: Optional[str] = None,           # e.g. "ibm_cleveland"
    api_key: Optional[str] = None,
    noise_model_from: Optional[str] = None,  # e.g. "ibm_cleveland"
)
```

### Pre-conditions

- `size` must satisfy `size > 0 and size & (size - 1) == 0`. If not,
  raises `ValueError("The vector dimensionality must be a power of 2.")`.
  Typical values are 32, 64, 128, 256, 512.
- `seed` must be an `int` (else `TypeError`).
- `vtype` is fixed to `"bipolar"` internally; you cannot configure it.

### Backend selection logic

| `channel` | Backend |
|:----------|:--------|
| `None` (default) | Local `AerSimulator` (`device="GPU"` if available, else CPU) |
| `None` + `noise_model_from="ibm_xxx"` | Local `AerSimulator` **with a noise model** harvested from that backend via `QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)` |
| Any non-None string | Live `QiskitRuntimeService(channel=channel, token=api_key, instance=instance)`; backend is `service.backend(backend)` or `service.least_busy(operational=True, simulator=False)` if `backend is None` |

If `noise_model_from` is supplied without `api_key`, raises
`ValueError("`api_key` must be provided to fetch backend properties for a noise model.")`.

### Stored attributes after `__init__`

| Attribute | Type | Notes |
|:----------|:-----|:------|
| `size`, `levels`, `shots`, `seed`, `vtype` | as configured |  |
| `level_hvs` | `list[np.ndarray]` | filled by `fit` |
| `prototypes` | `dict[str, QuantumCircuit]` | per-class circuits, filled by `fit` |
| `classes_` | `list[str]` | sorted, filled by `fit` |
| `class_counts_` | `dict[str, float]` | sample count per class for retrain |
| `backend` | `qiskit.providers.backend.Backend` |  |
| `version` | `str` | hdlib version |

## Level vector generation (inside `fit`)

```python
def _generate_level_vectors(D, num_levels, rng):
    level_vectors = [rng.choice([-1, 1], size=D)]
    change = int(D / 2)
    next_level = int((D / 2 / num_levels))
    for i in range(1, num_levels):
        prev_vec = level_vectors[i-1].copy()
        if i - 1 == 0:
            flip_indices = rng.choice(D, size=change, replace=False)
        else:
            flip_indices = rng.choice(D, size=next_level, replace=False)
        prev_vec[flip_indices] *= -1
        level_vectors.append(prev_vec)
    return level_vectors
```

This is the same similarity-preserving construction as the classical
classifier, but the resulting level vectors are kept as `np.ndarray` and
never inserted into a `Space`. They are used at encode time to look up
"which level does this feature value fall into".

## `fit(train_points, train_labels)`

```python
model.fit(train_points, train_labels)
```

- `train_points: list[list[float]]` &mdash; each inner list is a sample.
- `train_labels: list[str]` &mdash; one label per sample.

The function:

1. Generates `self.levels` level vectors (size `self.size`).
2. Iterates over each sample, for each feature index `i`:
   - Computes `level_index = max(0, min(L-1, int(value * (L-1))))`
     **assuming the feature value is in `[0, 1]`** &mdash; **scale your inputs first**.
   - Calls `quantum_encode(level_vec)` to make a phase oracle.
   - Calls `quantum_permute(qc, num_qubits, shift=i)` to rotate by the
     feature index.
3. Per class, bundles every sample's feature circuits via
   `quantum_bundle(...)` and `compress_circuit(...)`. Names the result
   `f"Prototype_{c}"` and stores it in `self.prototypes[c]`.
4. Stores `len(class_samples)` in `self.class_counts_[c]` for retrain.

After `fit`, the model holds one compressed circuit per class plus the
level-vector cache.

## `predict(test_points)`

```python
predictions, similarities = model.predict(test_points)
```

For each test sample:

1. Encode the sample into a list of feature circuits (same logic as `fit`).
2. Bundle them via `quantum_bundle(...)` and compress.
3. Run `run_compute_uncompute_test([query_circuit], prototype_circuits, backend, shots=..., seed=..., sampler=...)`.
4. Predict the class with the highest similarity (argmax).

On hardware:

- A single `Session(backend=self.backend)` covers the whole prediction call.
- Queries are batched in groups of 5 to avoid IBM's per-job memory cap.
- A `Sampler` is built with `dynamical_decoupling` (sequence `XpXm`) and
  `gate twirling` enabled.

Returns:

- `predictions: list[str]` &mdash; same order as `test_points`.
- `similarities: list[list[float]]` &mdash; per-sample similarity vector
  (one value per class, in `self.classes_` order).

Raises `RuntimeError("You must call fit before calling predict.")` if you
skipped `fit`.

## `retrain(train_points, train_labels, epochs=10, lr=1.0)`

Perceptron-style update: for each misclassified training sample at epoch
boundary, add the sample circuit to the **true** class prototype and
subtract it from the **predicted** class prototype, scaled by
`lr / class_counts_[label]` so that updates remain proportional to the
class mass.

Internally:

1. Run `self.predict(train_points)` once to bootstrap.
2. Snapshot the current prototypes as the best so far.
3. For up to `epochs` epochs:
   a. Encode every misclassified sample.
   b. For each misclassification, compute scaled phases via
      `_scale_circuit_phases` (positive factor for the true class,
      negative for the predicted class).
   c. Compose those updates onto each prototype and re-`compress_circuit`.
   d. Run `predict` again and compute the training error.
   e. If the error did not improve, revert to the snapshot and break.

Prints `epoch N: error_rate` for each epoch.

Returns `(best_error, final_epoch)`.

## Worked example &mdash; simulator end-to-end

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from hdlib.model import QuantumClassificationModel

iris = load_iris()
X, y = iris.data, iris.target

# Scale features to [0, 1] - REQUIRED for the quantum encoder
X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X.tolist(), y.tolist(), test_size=0.3, random_state=0, stratify=y
)

model = QuantumClassificationModel(size=32, levels=10, seed=42, shots=1024)
model.fit(X_train, y_train)

predictions, similarities = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, predictions))

# Optional: retrain for up to 10 epochs
final_error, last_epoch = model.retrain(X_train, y_train, epochs=10, lr=1.0)
print(f"final training error: {final_error}  (stopped at epoch {last_epoch})")

predictions, _ = model.predict(X_test)
print("post-retrain accuracy:", accuracy_score(y_test, predictions))
```

## Running on IBM Quantum hardware

```python
model = QuantumClassificationModel(
    size=32,
    levels=4,
    seed=42,
    shots=1024,
    channel="ibm_quantum_platform",
    instance="hub/group/project",
    backend="ibm_cleveland",
    api_key="...",
)
```

- A `Session` opens on `predict` for the duration of the call. Queries
  are batched into groups of 5.
- The sampler is configured with dynamical decoupling and gate twirling
  for robustness; you do not need to set those yourself.
- Mthree readout-error mitigation is applied automatically (the
  `M3Mitigation` is built from the backend properties before the first
  batch).

If you want simulator-with-noise (cheaper iteration than real hardware
but with realistic error rates):

```python
model = QuantumClassificationModel(
    size=32,
    levels=4,
    seed=42,
    noise_model_from="ibm_cleveland",
    api_key="...",
)
```

## Common pitfalls

- **`size` not a power of 2.** Raises `ValueError` at construction.
- **Features not in `[0, 1]`.** The level lookup
  `int(value * (L - 1))` only makes sense in that range. Scale your
  features first (e.g. `MinMaxScaler`).
- **Forgetting `fit` before `predict` / `retrain`** raises
  `RuntimeError`.
- **`size = 64` with `levels = 2`** wastes the quantum encoder. Use at
  least `levels = 4` to see meaningful classification.
- **Hardware sessions stay open for the full `predict` call**. Long
  test sets translate into long session usage; budget your IBM credits
  accordingly. Trim `test_points` to the minimum needed.
- **`retrain` prints progress to stdout** &mdash; capture or redirect if you
  call it from a notebook with assertion-style tests.
- **`AerSimulator(device="GPU")` requires a GPU-built `qiskit-aer`.**
  The constructor falls back to CPU silently if GPU init fails. Do not
  rely on GPU acceleration in CI without verifying.
- **Mismatched `levels` and `shots`**: too few `shots` leads to noisy
  similarity estimates; default `1024` is the lower bound for stable
  predictions. Try `4096` or `8192` if accuracy is unstable across runs.

## See also

- `hdlib-quantum-arithmetic` for the primitives (`encode`, `bundle`,
  `bind`, `permute`, `compress_circuit`).
- `hdlib-quantum-advanced` for the compute-uncompute test that powers
  `predict`, plus the more exotic operators.
- `hdlib-classification` for the classical sibling.
- `hdlib-installation` for setting up an IBM Quantum account.
