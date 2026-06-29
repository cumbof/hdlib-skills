---
name: hdlib-analogical-reasoning
description: Use when implementing role-filler binding analogies with hdlib (the canonical "What is the Dollar of Mexico?" Kanerva-style reasoning pattern). Covers building role and filler vectors, composing structured representations via bind + bundle, querying them with unbind, and using Space.find to recover the answer. Load whenever the user asks about VSA analogical reasoning, knowledge representation with hypervectors, or the dollar-of-Mexico problem.
---

# Analogical reasoning with hdlib

A canonical demonstration of VSA: encode two structured concepts as
**bundles of role-filler bindings**, derive an analogical mapping
between them, and recover one concept's slot value when given the
other's.

This is the "Dollar of Mexico" pattern from Pentti Kanerva's HDC
literature, ported to `hdlib` in `examples/quantum/reasoning.py`. The
underlying mechanic uses only `bind`, `bundle`, `Vector` and
`Space.find` &mdash; no models, no learning.

## When to use this skill

- The user asks to implement an analogy task with hyperdimensional
  vectors.
- The user wants a knowledge-representation tutorial using hdlib.
- The user wants to compute "X is to Y as A is to ?" with VSA.

## Core idea

Represent each entity as a **superposition of role-filler bindings**:

```
USA   = bundle( bind(NAM, usa), bind(CAP, wdc), bind(MON, dol) )
MEX   = bundle( bind(NAM, mex), bind(CAP, mxc), bind(MON, pes) )
```

Each `bind(ROLE, FILLER)` is a vector dissimilar to both `ROLE` and
`FILLER` but invertible: `bind(bind(ROLE, FILLER), ROLE) == FILLER`
because `bind` is its own inverse (XOR or component-wise product).

Bundle is similar to each of its inputs, so `USA.dist(bind(NAM, usa))`
is small, but `USA.dist(usa)` is large.

To answer "What is the dollar of Mexico?":

```
F_UM  = bind(USA, MEX)            # The analogical mapping vector
guess = bind(DOL, F_UM)
       = bind(DOL, bind(USA, MEX))
```

Because `bind` distributes (roughly) over `bundle`, this expansion
reveals that `guess` is close to `PES` (the peso slot in `MEX`) and
distant from every other symbol in the space.

## Recipe (pure classical hdlib)

```python
from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic import bind, bundle

D = 10000
space = Space(size=D, vtype="bipolar")

# Atomic concepts (role and filler vectors)
for name in ["USA", "DOL", "WDC",       # United States, Dollar, Washington DC
             "MEX", "PES", "MXC",       # Mexico, Peso, Mexico City
             "NAM", "CAP", "MON"]:      # Roles: name, capital, money
    space.insert(Vector(name=name, size=D, vtype="bipolar"))

def v(name):
    return space.get(names=[name])[0]

# Compose composite entities for each country
usa_record = bundle(bundle(
    bind(v("NAM"), v("USA")),
    bind(v("CAP"), v("WDC")),
), bind(v("MON"), v("DOL")))

mex_record = bundle(bundle(
    bind(v("NAM"), v("MEX")),
    bind(v("CAP"), v("MXC")),
), bind(v("MON"), v("PES")))

# Mapping vector: what does "US" map to in "Mexico"?
mapping = bind(usa_record, mex_record)

# Query: "What is the dollar of Mexico?"
guess = bind(v("DOL"), mapping)
guess.normalize()

# Search across the space for the closest symbol
name, dist = space.find(guess)
print(name, dist)        # expected: "PES" with a small distance
```

The same flow works for any "X is to A as Y is to ?" question:

- Replace `(usa_record, mex_record)` with whichever pair of records
  encode the source and target structures.
- Replace `v("DOL")` with the source slot value you want to translate.

## Choosing dimensionality

- `D = 1000` is the floor for reliable retrieval with 6 - 8 role-filler
  pairs. Below that, bundle capacity collapses.
- `D = 10000` (the hdlib default) gives comfortable headroom for the
  "Dollar of Mexico" task &mdash; the closest match is unambiguous.
- For more roles per record (10+), bump to `D = 32000` to keep distances
  well-separated.

## Variations and extensions

### Three-way analogies

```python
mapping_3 = bind(usa_record, bind(mex_record, japan_record))
```

works for short chains of bindings, but bundle capacity erodes with each
extra binding. Use it sparingly.

### Hierarchical records

Records can themselves be slot values:

```python
country_record = bundle(
    bind(v("NAM"), v("USA")),
    bind(v("REGION"), region_record),     # region_record is itself a bundle
)
```

Recover with iterated unbind:

```python
guessed_region = bind(country_record, v("REGION"))
guessed_region.normalize()
# Then search inside the region records' space
```

### Cleanup memory

If your query vector is noisy and `Space.find` returns the wrong symbol,
build a "cleanup" subset:

```python
cleanup = Space(size=D, vtype="bipolar")
for n in ["DOL", "PES", "WDC", "MXC"]:    # candidate fillers only
    cleanup.insert(space.get(names=[n])[0])

best, _ = cleanup.find(guess)
```

This restricts `find` to the legitimate slot-value candidates and gives
a cleaner answer.

## Quantum variant

`examples/quantum/reasoning.py` reproduces the same reasoning on quantum
oracles using `hdlib.arithmetic.quantum.encode`, `bundle`, and
`bind`, plus `run_compute_uncompute_test` to compute similarity. The
algebra is identical; the implementation swaps each `Vector` for a
`QuantumCircuit`. See `hdlib-quantum-arithmetic` and
`hdlib-quantum-advanced`.

## Common pitfalls

- **Forgetting to normalise** before searching with `Space.find` &mdash; the
  composite `guess` vector typically lands in `{-3, -1, 1, 3}` or wider.
  Cosine distance still works without normalisation, but the absolute
  numbers are misleading and other models that assume `{-1, +1}` will
  misbehave.
- **Using the same role across multiple records.** `bind(NAM, USA)` and
  `bind(NAM, MEX)` share the role vector; that is intentional. The
  composite records remain separable because the filler differs and
  `bundle` is content-addressable. But within a *single* record, every
  role must be unique &mdash; otherwise the binding interferes with itself.
- **Bundle overflow.** Stuffing more than ~30 - 50 bindings into one
  record at `D = 10000` makes retrieval unreliable. Use larger `D` or
  hierarchical decomposition.
- **Mismatched vtype.** All vectors involved must share `vtype`. Mix
  binary and bipolar and you get `Exception("Vector types are not compatible")`.
- **Naming collisions in the space.** Use a clear naming convention so
  fillers and roles are distinguishable (e.g. uppercase 3-letter codes
  for atoms, prefixes for composites).

## See also

- `hdlib-arithmetic` for the algebraic properties of `bind` and `bundle`.
- `hdlib-vectors` and `hdlib-space` for the building blocks.
- `hdlib-distance` for picking a threshold when retrieval is ambiguous.
- `examples/quantum/reasoning.py` in the hdlib repo for the quantum
  version.
