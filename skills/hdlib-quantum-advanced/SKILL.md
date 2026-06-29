---
name: hdlib-quantum-advanced
description: Use when invoking the advanced Quantum HDC primitives in hdlib.arithmetic.quantum that go beyond MAP arithmetic - the compute-uncompute similarity test (run_compute_uncompute_test), superposition_bundle (LCU-style parallel bundling), entangled_bind (SWAP-test entangled records), grover_search (amplitude amplification over a codebook), quantum_majority_bundle, quantum_contextual_bind, get_circuit_metrics, and the M3 readout error mitigation pipeline. Load whenever the user reaches for any of those functions, asks about quantum-only HDC algorithms, or wants to estimate similarities / search a codebook on a quantum backend.
---

# Advanced quantum HDC primitives

`hdlib.arithmetic.quantum` exposes a set of "post-MAP" operations that
exploit quantum-only features: parallel oracle evaluation, entanglement
between hypervectors, Grover-style amplification, and similarity
estimation via compute-uncompute. They are the building blocks of
`QuantumClassificationModel` and of higher-level QHDC algorithms.

This skill complements `hdlib-quantum-arithmetic`, which covers the
classical-MAP-on-circuits primitives (encode/bundle/bind/permute).

## When to use this skill

- The user imports `run_compute_uncompute_test`, `superposition_bundle`,
  `entangled_bind`, `grover_search`, `quantum_majority_bundle`,
  `quantum_contextual_bind`, or `get_circuit_metrics` from
  `hdlib.arithmetic.quantum`.
- The user wants to measure `|<L|R>|` or `|<L|R>|^2` similarity between
  two oracle circuits on a quantum backend.
- The user implements an HDC nearest-neighbour search on a quantum
  device.
- The user needs circuit-level depth / CNOT counts after transpilation.

## Import surface

```python
from hdlib.arithmetic.quantum import (
    run_compute_uncompute_test,
    superposition_bundle,
    entangled_bind,
    grover_search,
    quantum_majority_bundle,
    quantum_contextual_bind,
    get_circuit_metrics,
    compress_circuit,           # frequently used with the above
    encode,                     # to build oracle circuits
    statevector_to_bipolar,
)
```

## `run_compute_uncompute_test`

Estimates `|<L|R>|` between two oracle circuits without controlled gates.
Used as the similarity metric inside `QuantumClassificationModel.predict`.

```python
def run_compute_uncompute_test(
    state_left_circs: list[QuantumCircuit],
    state_right_circs: list[QuantumCircuit],
    backend: qiskit.providers.backend.Backend,
    shots: int = 1024,
    seed: int = 42,
    sampler: Optional[qiskit_ibm_runtime.Sampler] = None,
) -> tuple[list[list[float]], list[dict]]:
    ...
```

For each pair `(L_i, R_j)` it builds a tiny circuit:

1. `H` on every system qubit -> uniform superposition.
2. Apply `R_j` (compute).
3. Apply `L_i.inverse()` (uncompute).
4. `H` on every system qubit -> phase back to amplitudes.
5. Measure.

The probability of getting the all-zero outcome equals `|<L|R>|^2`. The
function returns the **square root** (`|<L|R>|`) so consumers can treat
the value as a similarity directly.

### Behaviour by backend

- **`AerSimulator`** &mdash; the function transpiles with
  `optimization_level=1` and runs the batch directly. No mitigation,
  uses `seed_simulator=seed` for reproducibility.
- **Hardware** (`backend` is not `AerSimulator`) &mdash; requires a `Sampler`
  argument. Transpiles with `optimization_level=3`, runs through the
  sampler, then applies **mthree readout-error mitigation** via
  `M3Mitigation(backend)` calibrated on the union of measured physical
  qubits across the batch.

### Return shape

`similarities[i][j]` is the estimated `|<L_i|R_j>|`. `counts[k]` are the
underlying raw / mitigated counts for the k-th query-prototype pair
(flattened in row-major order).

### Errors

- Mismatched `num_qubits` between left and right circuits raises
  `ValueError`.
- Hardware execution without a `sampler` raises `ValueError`.

### Usage example

```python
import numpy as np
from qiskit_aer import AerSimulator
from hdlib.arithmetic.quantum import encode, run_compute_uncompute_test

backend = AerSimulator()

queries     = [encode(np.array([1, -1, 1, -1, 1, -1, 1, -1]))]
prototypes  = [encode(np.array([ v, -v, v, -v, v, -v, v, -v])) for v in (1, -1)]

sims, _ = run_compute_uncompute_test(queries, prototypes, backend, shots=2048, seed=0)
print(sims)        # [[~1.0, ~0.0]] -> matches prototype 0
```

## `get_circuit_metrics(circuit, num_system_qubits, backend, optimization_level=3)`

Transpiles `circuit` to the backend's native gates and reports the most
common cost metrics:

```python
metrics = get_circuit_metrics(qc, num_system_qubits=3, backend=AerSimulator())
print(metrics)
# {
#   "num_qubits_total": ...,
#   "num_qubits_system": 3,
#   "num_qubits_ancilla": ...,
#   "depth": ...,
#   "cnot_count": ...,      # sums cx + ecr + cz (IBM may emit ecr natively)
#   "ops_count": OrderedDict(...),
# }
```

Raises `ValueError` if `num_system_qubits > circuit.num_qubits`.

Use it to benchmark `compress_circuit` versus uncompressed circuits, or
to compare different bundling strategies.

## `superposition_bundle(circuits)`

A **single** quantum SELECT (LCU) unitary that bundles `N` oracle
circuits in parallel.

```python
bundled_circ = superposition_bundle([encode(v) for v in vectors])
```

What it does (conceptually):

- Uses an index register of `n_idx = ceil(log2(N))` qubits placed in
  uniform superposition.
- Applies each `O_k` controlled on `|k>` in the index register.
- After the final Hadamard on the index register, the amplitude in the
  `|0...0>` index subspace encodes the element-wise sum of all input
  oracle vectors.

In code, the function builds the SELECT internally
(`_build_select_circuit`), simulates with `Statevector`, decodes via
`_decode_select_bundle` (removing the bias from unused index slots if
`N` is not a power of 2), and **re-encodes** as a single phase oracle.
The returned circuit has the same shape as `encode(...)` output, ready
for further composition.

Empty input list raises `ValueError`.

> **Practical note:** because `superposition_bundle` simulates classically
> at the moment, you don't actually realise the depth advantage on
> hardware &mdash; but the resulting *output* oracle is shallow and runnable
> anywhere.

## `quantum_majority_bundle(circuits, backend=None, shots=1024)`

Same idea as `superposition_bundle` but explicitly named for the
**majority vote** interpretation. The result equals what you would get
from the classical `bundle` followed by `.normalize()` on every basis
state.

```python
from hdlib.arithmetic.quantum import quantum_majority_bundle, statevector_to_bipolar
majority = quantum_majority_bundle([encode(v) for v in five_vectors])
print(statevector_to_bipolar(majority))   # one +/-1 per basis state
```

Empty list raises `ValueError`. `backend` and `shots` arguments are
reserved for a future sampling-based decode; currently the function uses
the same statevector path as `superposition_bundle`.

## `entangled_bind(circuit1, circuit2)`

Builds a `(2n + 1)`-qubit register `(anc, sys_a, sys_b)` and prepares:

|Phi> = 1/sqrt(2) * (|0>|psi_1>|psi_2> + |1>|psi_2>|psi_1>)

using a SWAP-test construction (`H` on the ancilla, then `cswap` on
every paired sys_a/sys_b qubit). Measuring the ancilla in the Hadamard
basis yields the SWAP-test outcome distribution:

- `P(|0>) = (1 + |<psi_1|psi_2>|^2) / 2`
- `P(|1>) = (1 - |<psi_1|psi_2>|^2) / 2`

The two system registers remain entangled and can be unrolled later.

Raises `ValueError` if the two circuits have different qubit counts.

### Use cases

- Reversible binding: classical `bind` is irreversible (XOR collapses
  information); `entangled_bind` keeps the components recoverable via
  unitary uncomputation.
- Similarity-as-side-effect: the SWAP test's ancilla probability is the
  similarity itself.

```python
from hdlib.arithmetic.quantum import encode, entangled_bind
v1 = np.array([1, -1, 1, -1])
v2 = np.array([-1, 1, -1, 1])
qc = entangled_bind(encode(v1), encode(v2))
print(qc.num_qubits)   # 5 = 1 (anc) + 2 (sys_a) + 2 (sys_b)
```

## `grover_search(query_circuit, codebook_circuits, similarity_threshold=0.8, backend=None, shots=1024)`

Two-stage nearest-neighbour search over a codebook of oracle circuits.

```python
best_idx, best_sim = grover_search(query_circuit, codebook_circuits,
                                   similarity_threshold=0.8,
                                   backend=AerSimulator(),
                                   shots=2048)
```

Stages:

1. **Similarity estimation.** Calls `run_compute_uncompute_test([query], codebook)`
   to estimate `|<query|codebook[k]>|` for every `k`.
2. **Grover amplification.** Marks indices whose similarity exceeds
   `similarity_threshold` (falls back to the single best if nothing
   passes), then runs `n_iter = round(pi / (4 * sqrt(N / |marked|)) - 0.5)`
   Grover iterations on the index register and measures.

Returns `(best_index, best_sim)`. `best_index` is the most-frequently
measured index from the Grover step (clamped to `[0, N)`); `best_sim`
is the similarity of that index from stage 1.

Raises `ValueError` if `codebook_circuits` is empty.

> **Note:** the similarity step uses simulation, so the end-to-end
> demonstrates the algorithm structure rather than a quantum speedup.
> The Grover iterations themselves are real quantum work.

## `quantum_contextual_bind(context_circuit, value_circuits)`

Creates a register `(idx, sys)` of `(n_idx + n_sys)` qubits encoding the
superposition of all `K` bindings simultaneously:

|psi_ctx> = 1/sqrt(K) * sum_k |k> tensor |bind(context, v_k)>

Measuring the index register in state `|k>` projects the system register
onto the specific binding `bind(context, v_k)` &mdash; a quantum key-value
lookup. Useful as a substrate for Grover-style retrieval over a
context-indexed codebook.

```python
context = encode(np.random.choice([-1, 1], size=4))
values  = [encode(np.random.choice([-1, 1], size=4)) for _ in range(4)]
ctx_circuit = quantum_contextual_bind(context, values)
print(ctx_circuit.num_qubits)   # n_idx (2) + n_sys (2) = 4
```

Raises `ValueError` for empty value list or mismatched qubit counts.

## Common pitfalls

- **`run_compute_uncompute_test` on hardware without a `Sampler`** &mdash;
  raises `ValueError`. Build the `Sampler` from your `Session` and pass
  it as `sampler=...`.
- **Memory blowup with `superposition_bundle` for large `N`.** The
  function uses `_decode_select_bundle` which performs a full
  statevector simulation of the `(n_idx + n_sys)`-qubit circuit. For
  `n_sys = 10, N = 1024` you simulate 2^20 amplitudes; fine. For
  `n_sys = 14, N = 16` you simulate 2^18 amplitudes; still fine.
  For larger sizes consider chunking.
- **`grover_search` without enough marked items** &mdash; the function
  falls back to a single marked element (the best), so Grover degenerates
  to a single oracle / diffusion iteration. Increase the codebook or
  lower `similarity_threshold`.
- **mthree readout mitigation can fail to converge** on poorly
  characterised hardware backends; if `cals_from_system` throws, fall
  back to simulator or pass a pre-built mitigator.
- **Indexing convention.** `run_compute_uncompute_test` returns
  `similarities[i][j]` indexed by `(query, prototype)` &mdash; not the other
  way around. The `QuantumClassificationModel.predict` iterates queries
  before prototypes for this reason.
- **Vector sizes are power-of-2 only.** All advanced primitives inherit
  this constraint from `encode`. Pad with `+1` only if you accept the
  bias.

## See also

- `hdlib-quantum-arithmetic` &mdash; the basic encode / bundle / bind /
  permute / compress_circuit primitives every advanced operator builds on.
- `hdlib-quantum-classification` &mdash; uses `run_compute_uncompute_test`
  as its similarity backbone.
- `hdlib-installation` for IBM Quantum credential setup.
