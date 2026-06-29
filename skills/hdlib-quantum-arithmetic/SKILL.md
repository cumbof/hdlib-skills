---
name: hdlib-quantum-arithmetic
description: Use when building Quantum Hyperdimensional Computing primitives with hdlib.arithmetic.quantum - encoding a bipolar vector as a phase oracle (encode), bundling / binding / permuting on quantum circuits (bundle, bind, permute), compressing deep oracles into a single DiagonalGate (compress_circuit), or decoding a circuit back to a classical bipolar vector (statevector_to_bipolar). Load whenever the user imports from hdlib.arithmetic.quantum or constructs Qiskit circuits to represent hypervectors.
---

# Quantum MAP arithmetic with `hdlib.arithmetic.quantum`

`hdlib.arithmetic.quantum` is the bridge between hdlib's classical
bipolar vectors and Qiskit `QuantumCircuit`s. A hypervector becomes a
**phase oracle**: a diagonal gate that applies `+1` or `-1` (i.e. phase
`0` or `pi`) to each computational basis state. Once vectors are in
circuit form, the module reproduces every classical MAP operator on the
circuit level and adds operators that only make sense quantumly
(superposition bundle, entangled bind, Grover search, ...).

## When to use this skill

- The user imports anything from `hdlib.arithmetic.quantum`.
- The user constructs a quantum hypervector circuit from a bipolar
  ndarray.
- The user is preparing the input to `QuantumClassificationModel` or the
  more advanced primitives (`hdlib-quantum-advanced`).
- The user wants to decode a quantum oracle back into a `numpy.ndarray`.

## Constraints (these apply to every function below)

- **Vector size must be a power of 2.** `encode` pads with `+1` if it is
  not exactly `2 ** n`, but downstream operations assume the padded size.
  Authoring bipolar vectors with `size in {16, 32, 64, ..., 1024}` is
  standard for QHDC.
- **Vector type must be bipolar.** Binary vectors are not supported.
- **Qiskit version**: `qiskit >= 2.2.1` (matches `hdlib`'s requirements).
- Functions raise either `ValueError` or `TypeError` on bad input &mdash;
  there is no shared base exception type.

## Import surface

```python
from hdlib.arithmetic.quantum import (
    encode,                  # bipolar ndarray -> QuantumCircuit (phase oracle)
    bind,                    # list[QuantumCircuit] -> bound QuantumCircuit
    bundle,                  # list[QuantumCircuit] -> bundled QuantumCircuit
    permute,                 # QuantumCircuit, num_qubits, shift -> permuted circuit
    compress_circuit,        # deep circuit -> single DiagonalGate
    statevector_to_bipolar,  # QuantumCircuit -> {-1, +1} ndarray
)
```

## `encode(vec_bipolar, label="O_v")`

```python
import numpy as np
from hdlib.arithmetic.quantum import encode

vec = np.array([1, -1, 1, 1, -1, -1, 1, -1])   # size 8 == 2**3
qc = encode(vec, label="O_v")                  # 3-qubit phase oracle
```

- Maps the `i`-th component to the diagonal entry of a `DiagonalGate`:
  `+1` -> phase `0`, `-1` -> phase `pi`.
- Pads `vec` with `+1` if `len(vec)` is not `2**n`; the implicit
  `num_qubits = ceil(log2(len(vec)))`.
- Returns a `QuantumCircuit` whose only operation is the `DiagonalGate`.
- Raises `ValueError` if the input contains anything other than `-1` and
  `+1`.

## `statevector_to_bipolar(circuit)`

The inverse: simulate the oracle on `|+>^n`, read the resulting
statevector, and turn the real parts back into `{-1, +1}` integers.

```python
from hdlib.arithmetic.quantum import statevector_to_bipolar
recovered = statevector_to_bipolar(qc)
assert (recovered == vec).all()
```

- Internally applies `H` gates to every qubit, composes with the input
  circuit, simulates, and inspects `np.real(state.data)`.
- Detects whether the state is in **standard encoding** (`0` / `pi`) or
  **symmetric encoding** (`+/- delta`); rotates by `-90deg` in the latter
  case to project the phases back to real values.
- Returns an `np.ndarray` of `int` with one element per basis state.

You should expect this function to be used at the *end* of a quantum
pipeline, not after every operation: each `statevector_to_bipolar`
performs a small simulation.

## `bind(circuits)`

```python
qc_bind = bind([qc1, qc2, qc3])
```

- Sequentially composes the circuits via `qc.compose(circ, inplace=True)`.
- All inputs must have the same `num_qubits` (else `ValueError`).
- Empty list raises `ValueError("Input list for bind cannot be empty.")`.

This is "phase XOR": composing two phase oracles adds their phases mod
`2pi`. Two `-1` phases (`pi + pi = 2pi`) cancel to `0` &mdash; equivalent to
classical bind being an XOR / multiplication.

> **Composability limit:** binding two **symmetric** bundles (outputs of
> `bundle(..., method="average")`) is not exact &mdash; the docstring warns
> about this. Use `bind` on freshly `encode`'d oracles or on
> `compress_circuit` outputs.

## `bundle(circuits, method="average")`

Bundles multiple oracles into a single one whose phase encodes the
majority vote of the inputs.

- `method="average"` (default) &mdash; the symbolic phase-accumulation
  variant. For each basis state `j`, computes a vote (`+1` or `-1`)
  across all input circuits, then scales the symmetric target phase by
  `1 / N` (the number of inputs). Preserves composability for further
  bundling.
- `method="classical"` &mdash; falls back to the classical implementation:
  it decodes every input to a bipolar ndarray, computes the element-wise
  sum, maps via `exp(i * pi * x / rms)` to a unit-modulus phase vector,
  and re-encodes as a single `DiagonalGate`. Use this when you want exact
  arithmetic and do not need to chain further quantum operations.

```python
bundled_qc = bundle([encode(v) for v in many_vectors])
# or:
bundled_qc = bundle([encode(v) for v in many_vectors], method="classical")
```

Empty list raises `ValueError`.

## `permute(qc, num_qubits, shift=0)`

Cyclically rotates the basis-state labels by `shift` positions modulo
`2 ** num_qubits`. Implemented as a series of MCX-gate "binary
incrementers" so the result is an O(num_qubits) digital circuit, not a
SWAP network.

```python
shifted = permute(qc, num_qubits=3, shift=2)   # rotate basis states by 2
```

- `shift=0` returns the input unchanged.
- `shift` is reduced modulo `2 ** num_qubits`.
- Passing `qc=None` produces a permutation-only circuit:
  `permute(None, num_qubits=3, shift=1)`.

This is the quantum counterpart of `numpy.roll`.

## `compress_circuit(circuit)`

Computes the exact noise-free statevector that the input circuit would
produce on `|+>^n`, extracts the relative phases, and synthesises a
single shallow circuit consisting of a Hadamard layer plus one
`DiagonalGate` with those phases. The result is mathematically identical
to the input but a few orders of magnitude shallower &mdash; essential for
hardware execution.

```python
shallow = compress_circuit(deep_qc)
```

Use this after every `bundle` / `bind` round when you want to:

- Keep the circuit depth manageable on real hardware.
- Re-use the result as a fresh oracle for further composition.

> **Quantum advantage caveat:** `compress_circuit` simulates classically
> &mdash; it is not a quantum shortcut, just a circuit-level optimisation.
> The depth it produces *is* a real advantage at execution time.

## Full quantum MAP recipe

```python
import numpy as np
from hdlib.arithmetic.quantum import (
    encode, bundle, bind, permute,
    compress_circuit, statevector_to_bipolar,
)

# Three bipolar vectors of size 8 (size 2**3)
v1 = np.array([ 1,-1, 1, 1,-1,-1, 1,-1])
v2 = np.array([-1, 1, 1,-1, 1,-1,-1, 1])
v3 = np.array([ 1, 1,-1,-1, 1, 1,-1,-1])

c1, c2, c3 = encode(v1), encode(v2), encode(v3)

# Classical bundle inside quantum
bundled = bundle([c1, c2, c3], method="classical")
print(statevector_to_bipolar(bundled))    # majority vote over v1, v2, v3

# Binding
bound = bind([c1, c2])
print(statevector_to_bipolar(bound))      # element-wise product of v1 and v2

# Permutation
shifted = permute(c1, num_qubits=3, shift=2)
print(statevector_to_bipolar(shifted))    # np.roll(v1, 2)

# Compose, then compress for hardware
deep = bind([bundled, c3])
shallow = compress_circuit(deep)
print(shallow.depth(), shallow.size())   # both small
```

## What lives in this module vs the classical one

| Operation | Classical (`hdlib.arithmetic`) | Quantum (`hdlib.arithmetic.quantum`) |
|:----------|:-------------------------------|:--------------------------------------|
| Construct a hypervector | `Vector(...)` | `encode(np.ndarray)` |
| Bundle | `bundle(v1, v2)` &mdash; pairwise on Vector | `bundle([qc1, qc2, ...])` &mdash; list of circuits |
| Bind | `bind(v1, v2)` &mdash; pairwise on Vector | `bind([qc1, qc2, ...])` &mdash; list of circuits |
| Subtract | `subtraction(v1, v2)` | not exposed (use `bind` for XOR-style inverse) |
| Permute | `permute(v, rotate_by)` | `permute(qc, num_qubits, shift)` |
| Distance | `Vector.dist` | `run_compute_uncompute_test` (see `hdlib-quantum-advanced`) |

## Common pitfalls

- **Vector size not a power of 2.** `encode` pads silently with `+1`,
  which changes downstream similarity calculations. Always allocate
  bipolar vectors with `size in {16, 32, 64, 128, ...}`.
- **Bundling a single circuit** is a no-op (the result equals the input).
  The function does not raise but you probably meant to bundle several.
- **`bind` on already-bundled circuits with `method="average"`** does not
  preserve composability past one round. Use `compress_circuit` between
  rounds, or use `method="classical"` for the final bundle.
- **`statevector_to_bipolar` decompiles the entire circuit** &mdash; it is
  O(2^n) memory and time. Avoid calling it after every step in a long
  pipeline; use it only at the boundary back to classical code.
- **Mixing classical and quantum sizes.** `D = 10000` is fine classically
  but completely impractical quantumly (`log2(10000) ~= 14` qubits is
  fine; `D = 2^14 = 16384` is fine; `D = 10000` is *not* a power of 2).
  Plan vector sizes accordingly.

## See also

- `hdlib-arithmetic` for the classical counterparts.
- `hdlib-quantum-classification` for the classifier that builds on these
  primitives.
- `hdlib-quantum-advanced` for compute-uncompute similarity,
  superposition bundle, entangled bind, Grover search, and circuit
  metrics.
