---
name: hdlib-arithmetic
description: Use when the user calls bundle, bind, subtraction, or permute on hyperdimensional vectors with hdlib (the classical MAP arithmetic). Covers function semantics, algebraic properties (invertibility, similarity preservation), the difference between the module-level functions in hdlib.arithmetic (which return new Vectors) and the Vector methods (which mutate in place), normalization after composition, and binary-vs-bipolar behaviour. Load whenever the user reaches for the `+`, `-`, `*` operators on Vector objects or imports anything from hdlib.arithmetic.
---

# Classical MAP arithmetic in hdlib

`hdlib.arithmetic` implements the **MAP** (Multiply / Add / Permute)
arithmetic that defines Vector-Symbolic Architectures. Four functions
make up the API:

| Function | Symbolic role | Operator |
|:---------|:--------------|:---------|
| `bundle(v1, v2)` | superposition / "and-also" | `v1 + v2` |
| `bind(v1, v2)` | role-filler / "for the role X, the value is Y" | `v1 * v2` |
| `subtraction(v1, v2)` | inverse of bundling (bipolar only) | `v1 - v2` |
| `permute(v, rotate_by=1)` | sequencing / position encoding | (none) |

## When to use this skill

- The user imports anything from `hdlib.arithmetic`.
- The user uses `+`, `-`, `*` on `Vector` objects.
- The user calls `Vector.bind`, `Vector.bundle`, `Vector.subtraction`, or
  `Vector.permute`.
- The user composes multiple vectors and asks about normalisation,
  invertibility, or similarity properties.

## Import

```python
from hdlib.arithmetic import bind, bundle, subtraction, permute
```

## API contract

Every function follows the same contract: it takes one or two `Vector`
objects and returns a **new** `Vector` whose:

- `size` matches the inputs.
- `vtype` matches the inputs.
- `tags` is the **union** of the inputs' tags (except `subtraction`, which
  inherits the first operand's tags only).
- `seed` is taken from the first operand (useful for reproducibility
  hashing but not actually used to regenerate the data).
- `name` is auto-generated (UUID v4) &mdash; rename it explicitly before
  inserting into a `Space`.

The corresponding `Vector` instance methods (`v1.bind`, `v1.bundle`, ...)
do the same computation but **mutate `v1` in place** via `__override_object`.
Use the module functions when you want pure functions; use the methods
when you want to chain mutating operations.

### `bind(v1, v2)` &mdash; element-wise product / XOR

Bipolar: `result = v1.vector * v2.vector` (each component is `+1 * +1`,
`+1 * -1`, etc.; the result stays in `{-1, +1}`).

Binary: `result = v1.vector.astype(int) ^ v2.vector.astype(int)` (XOR).

Raises `Exception` if either size or vtype differs.

#### Properties

- **Invertible.** `bind(bind(v1, v2), v2) == v1` (because `x * x == 1` and
  `x XOR x == 0`). Used for unbinding role-filler pairs.
- **Distributes over bundling.** `bind(a, bundle(b, c)) approx bundle(bind(a, b), bind(a, c))`
  (exact for `XOR`, approximate after bundling normalises).
- **Preserves distance.** `bind(a, x).dist(bind(b, x)) approx a.dist(b)`.
- **Dissimilar to inputs.** `bind(a, b)` is near-orthogonal to both `a`
  and `b` &mdash; useful for creating a fresh symbol while remembering its
  components.

### `bundle(v1, v2)` &mdash; element-wise sum / majority

Bipolar: `result = v1.vector + v2.vector` &mdash; components may be in
`{-2, 0, 2}` (because each addend is `+/-1`). The result is **not**
normalised; call `.normalize()` on the returned Vector to project back to
`{-1, +1}`.

Binary: `result = ((v1.vector + v2.vector) > 1).astype(int)` &mdash; a
two-input majority vote (`AND` for two inputs).

Raises `Exception` if either size or vtype differs.

#### Properties

- **Similar to inputs.** `bundle(a, b).dist(a)` is markedly smaller than 1.
- **Capacity is finite.** Bundling many vectors degrades the similarity of
  each component to the bundle &mdash; once you have many vectors, none of
  them are individually retrievable from a single bundled vector.
- **Multiplicity matters.** Bundling the same vector multiple times pulls
  the bundle closer to it. Useful for weighted superpositions.

### `subtraction(v1, v2)` &mdash; bipolar only

Bipolar: `result = v1.vector - v2.vector` &mdash; components in `{-2, 0, 2}`.

Binary: raises `Exception("Subtraction is not available for binary vectors")`.

Other failure modes match `bundle` / `bind` (size, vtype).

Typically used to *remove* a vector from a bundle:

```python
mixture = a + b + c
without_b = mixture - b
without_b.normalize()           # snap back to {-1, +1}
```

After subtraction the resulting vector should usually be `normalize()`d
before further use.

### `permute(v, rotate_by=1)`

Implemented as `np.roll(v.vector, rotate_by, axis=0)`. Returns a new
`Vector` with the same data shifted by `rotate_by` positions (negative
values rotate the other way).

#### Properties

- **Invertible.** `permute(permute(v, k), -k) == v`.
- **Distributes over bundling and any element-wise operation.**
  `permute(bundle(a, b), k) == bundle(permute(a, k), permute(b, k))`.
- **Preserves distance.** Rotating both operands preserves their distance.
- **Dissimilar to inputs.** For non-trivial `k`, the rotated vector is
  near-orthogonal to the original.

Primary use: encoding sequence position. The `ClassificationModel` permutes
each level vector by the feature index before bundling them.

## Functions vs methods

```python
from hdlib.vector import Vector
from hdlib.arithmetic import bundle

v1 = Vector(name="v1", seed=1)
v2 = Vector(name="v2", seed=2)

# Module function -> new Vector
v3 = bundle(v1, v2)             # v1 and v2 untouched
v3.name = "v3"                  # default name is a UUID

# Vector method -> in-place mutation
v1.bundle(v2)                   # v1 is now bundle(old_v1, v2); name/tags merged
```

The module functions are pure (good for composition pipelines). The methods
are useful when you want to incrementally accumulate into one vector.

## Normalisation after composition

`bundle`, `subtraction`, and chained operations can leave the result with
components outside `{-1, +1}` (bipolar) or `{0, 1}` (binary). Call
`Vector.normalize()` to project:

- **Bipolar**: applies `np.sign` and replaces zeros with `+1`.
- **Binary**: divides by the maximum value and thresholds at `0.5`. Empty
  (all-zero) vectors stay all-zero.

```python
v_total = v1 + v2 + v3 + v4     # may contain values up to +/-4
v_total.normalize()             # now strictly +/-1
```

You do **not** need to normalise after `bind` or `permute`; both preserve
the original alphabet.

## Worked example &mdash; analogical reasoning ("Dollar of Mexico")

```python
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle

space = Space()
space.bulk_insert(names=[
    "USA", "DOL", "WDC",      # united states / dollar / washington dc
    "MEX", "PES", "MXC",      # mexico / peso / mexico city
    "NAM", "MON", "CAP",      # role vectors: name, money, capital
])

def get(*ns): return [space.get(names=[n])[0] for n in ns]

usa, dol, wdc, mex, pes, mxc, nam, mon, cap = get(
    "USA", "DOL", "WDC", "MEX", "PES", "MXC", "NAM", "MON", "CAP"
)

# Compose role-filler bindings for both countries
ustates = bundle(bundle(bind(nam, usa), bind(cap, wdc)), bind(mon, dol))
mexico  = bundle(bundle(bind(nam, mex), bind(cap, mxc)), bind(mon, pes))

# Define a transformation US -> Mexico
mapping = bind(ustates, mexico)

# "What is the dollar of Mexico?"  ->  bind dol with the mapping
guess = bind(dol, mapping)
guess.normalize()

# The closest symbol in the space should be the peso
print(space.find(guess))    # ("PES", small distance)
```

(The `hdlib-analogical-reasoning` skill explains the underlying VSA logic
and explores the role-filler pattern in more depth.)

## Common pitfalls

- **Forgetting to normalise.** A bundled vector with raw `{-3, -1, 1, 3}`
  values still works for `dist`, but storing it in a `Space` and treating
  it as a regular hypervector confuses downstream code. Normalise.
- **Binary subtraction.** Subtraction of binary vectors raises `Exception`.
  Use `bind` (XOR) if you mean "differing positions" in the binary case.
- **Tag explosion in repeated bundling.** Each bundle merges tags via
  `set.union`. After many bundles, your composite vector inherits every
  tag of every component. Strip or replace tags if your downstream code
  searches by tag.
- **Bundling many vectors degrades retrievability.** The fan-out limit
  depends on `size`; for `D=10000`, expect retrievable bundles up to
  ~50-100 component vectors before similarity collapses.
- **`permute(v, rotate_by=0)`** is a no-op identity. Permutations are
  effective at distinguishing positions only when `rotate_by != 0`.
- **Operator chaining returns intermediate `Vector` objects** with
  auto-generated UUID names &mdash; cosmetically noisy. Rename them before
  inserting into a space or wrap them in a helper.

## See also

- `hdlib-vectors` for the underlying `Vector` class.
- `hdlib-distance` for measuring similarity after composition.
- `hdlib-analogical-reasoning` for the canonical role-filler use case.
- `hdlib-quantum-arithmetic` for the corresponding quantum implementations.
- `hdlib-encoding-data` for how `bundle`, `bind`, and `permute` combine to
  encode tabular data into a single hypervector.
