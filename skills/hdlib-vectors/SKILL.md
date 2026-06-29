---
name: hdlib-vectors
description: Use when creating, combining, copying, comparing, or persisting hyperdimensional vectors with hdlib. Covers the hdlib.vector.Vector class - binary vs bipolar types, random initialization, deterministic seeds, operators (+/-/*), the dist / normalize / bind / bundle / subtraction / permute / dump methods, tags, parent/child links, and pickle round-trips. Load whenever the user writes Vector(...), v1 + v2, v1 * v2, vector.dist(...), or anything else that operates on a single hyperdimensional vector.
---

# `hdlib.vector.Vector` &mdash; the building block

The `Vector` class is the atomic unit of every hdlib program. It wraps a 1-D
`numpy.ndarray` of either `{0, 1}` (binary) or `{-1, +1}` (bipolar) entries,
together with a name, an optional set of tags, parent/child links, the seed
used to generate it, and the hdlib version that produced it.

## When to use this skill

- The user writes `Vector(...)`, `Vector(size=...)`, `Vector(from_file=...)`,
  or any other invocation that creates a hypervector.
- The user combines two vectors with `+`, `-`, `*`, calls `vector.dist(...)`,
  `vector.normalize()`, `vector.bind(...)`, `vector.bundle(...)`,
  `vector.subtraction(...)`, or `vector.permute(...)`.
- The user pickles a `Vector` with `dump`/`from_file`.
- The user is debugging a `TypeError` / `ValueError` raised from
  `hdlib/vector.py`.

## Import

```python
from hdlib.vector import Vector
# It is also re-exported from hdlib.space for convenience:
from hdlib.space import Vector  # equivalent
```

## Constructor

```python
Vector(
    name: Optional[str] = None,        # default: a fresh UUID v4 string
    size: int = 10000,                 # length of the underlying ndarray
    vector: Optional[np.ndarray] = None,  # provide explicit data
    vtype: str = "bipolar",            # "binary" or "bipolar"
    tags: Optional[Set[Union[str, int, float]]] = None,
    seed: Optional[int] = None,        # for reproducible random init
    warning: bool = False,             # print a version-skew warning on load
    from_file: Optional[str] = None,   # load a pickle dump instead of generating
)
```

### Three mutually exclusive construction modes

1. **Random generation (default).** Pass `size` and `vtype`. The constructor
   uses `numpy.random.default_rng(seed)` and produces a fresh random vector.
   If `seed` is `None` the rng is non-deterministic.
2. **Explicit data.** Pass `vector=<ndarray>`. `size` is then *derived* from
   `len(vector)`. The `vtype` you pass is honoured but **not validated**
   against the actual content of `vector`.
3. **Load from disk.** Pass `from_file=<path-to-pickle>`. All other arguments
   are ignored except `name` (it stays at whatever was pickled).

### Important defaults and invariants

- `vtype` is **`"bipolar"`** by default. The classical arithmetic and most
  models assume bipolar; switch to binary only when the user explicitly
  requests it.
- `size` defaults to **`10000`**. Any positive integer works; the
  ML literature converges on 1k - 10k for HDC.
- `name` is auto-generated as a string `uuid.uuid4()` if you do not pass one.
  Names are *cast to `str`* on assignment, so you can pass int / float and the
  Vector will store the string form.
- `tags` must be a `set` if provided. Each tag must be a primitive
  (`str`, `int`, `float`).
- `seed` must be an integer when provided (numpy's RNG requirement).
- The object stores `self.version = hdlib.__version__` so cross-version
  pickle loads can emit a warning.

### Raises

| Exception | Cause |
|:----------|:------|
| `TypeError` | `name` is not coercible to `str`; `tags` not a `set`; `vector` not an `np.ndarray`; `size` not an `int`; `seed` not an `int` |
| `ValueError` | `vtype` not in `{"binary", "bipolar"}` |
| `FileNotFoundError` | `from_file=...` path does not exist |
| `Exception` | The pickle at `from_file` is not a `Vector` instance |

## Random initialization details

Internally the constructor:

```python
rand = np.random.default_rng(seed)
self.vector = rand.integers(2, size=size)   # binary {0, 1}
if vtype == "bipolar":
    self.vector = 2 * self.vector - 1        # remap to {-1, +1}
```

Implications:
- Two random vectors of size `N` are nearly orthogonal (cosine distance
  approx 1.0) once `N >= 1000`.
- Components are integer-typed; the underlying dtype is whatever
  `np.random.Generator.integers` returns (typically `int64`).

## Attributes you can read or set

| Attribute | Type | Notes |
|:----------|:-----|:------|
| `vector.name` | `str` | Free to rename. The owning `Space` indexes by name, so do *not* rename a vector that is already inserted. |
| `vector.size` | `int` | Length of the ndarray. |
| `vector.vector` | `np.ndarray` | The raw data. Safe to read; write at your own risk. |
| `vector.vtype` | `str` | `"binary"` or `"bipolar"`. |
| `vector.seed` | `int | None` | The seed used for random init. |
| `vector.tags` | `set` | User tags. `Space.add_tag` keeps the space-level tag index in sync; setting this directly does not. |
| `vector.parents` | `set` | Names of parent vectors (set by `Space.link`). |
| `vector.children` | `set` | Names of child vectors. |
| `vector.version` | `str` | hdlib version at creation time. |

## Operators (return new `Vector` objects)

| Operator | Equivalent | Notes |
|:---------|:-----------|:------|
| `v1 + v2` | `hdlib.arithmetic.bundle(v1, v2)` | Element-wise sum (bipolar) or majority vote (binary) |
| `v1 - v2` | `hdlib.arithmetic.subtraction(v1, v2)` | Bipolar only &mdash; binary raises `Exception` |
| `v1 * v2` | `hdlib.arithmetic.bind(v1, v2)` | Element-wise product (bipolar) or XOR (binary) |

The operators raise `TypeError` if the right-hand side is not a `Vector`.
Mismatched `size` or `vtype` raise the more general `Exception`.

## In-place methods (mutate `self`)

```python
vector.bind(other)        # like v *= other, but for hdlib semantics
vector.bundle(other)      # like v += other
vector.subtraction(other) # bipolar only
vector.permute(rotate_by=k)
vector.normalize()        # collapse back to {-1, +1} or {0, 1}
```

Each of those imports the corresponding function from `hdlib.arithmetic`
internally and then `__override_object` copies the result back into `self`.
After bundling or binding many vectors, components can drift outside the
canonical range (e.g. a sum of three bipolar vectors lands in `{-3, -1, 1, 3}`).
Call `normalize()` to project back.

## Distance

```python
vector.dist(other, method="cosine")    # default
vector.dist(other, method="euclidean")
vector.dist(other, method="hamming")
```

- `cosine` &mdash; `1 - dot(v1, v2) / (||v1|| * ||v2||)`. The standard HDC metric.
- `hamming` &mdash; raw count of differing positions (integer).
- `euclidean` &mdash; `np.linalg.norm(v1 - v2)`.

Mismatched `size` or `vtype` between the two vectors raises `Exception`.
Unsupported `method` raises `ValueError`. See the `hdlib-distance` skill for
when to pick which metric.

## Persistence

```python
vector.dump(to_file="my_vector.pkl")        # absolute or relative path
loaded = Vector(from_file="my_vector.pkl")  # round-trip
```

- If `to_file` is omitted, the file is written into the current working
  directory as `<vector.name>.pkl`.
- `dump` raises `Exception` if the destination file already exists. It
  intentionally refuses to overwrite. The caller is responsible for using
  `os.remove(...)` or a fresh path.
- `from_file=...` calls `pickle.load` and copies the result's `__dict__`
  into `self`. If the pickle was produced by a *different* hdlib version
  the constructor prints a warning to stdout.

## Putting it together

### Recipe 1 &mdash; create two random vectors and compute their distance

```python
from hdlib.vector import Vector

v1 = Vector(name="v1", seed=1)
v2 = Vector(name="v2", seed=2)

print(v1.dist(v2))                  # ~0.996 (near orthogonal)
print(v1.dist(v2, method="hamming"))
```

### Recipe 2 &mdash; bundle three vectors and find which input dominates

```python
from hdlib.vector import Vector

red    = Vector(name="red",    seed=10)
green  = Vector(name="green",  seed=20)
blue   = Vector(name="blue",   seed=30)

mix = red + green + blue            # bundle, may drift outside {-1, +1}
mix.normalize()                     # snap back to {-1, +1}

for v in (red, green, blue):
    print(v.name, mix.dist(v))
```

All three distances will be close to 0.6 because every input contributed
equally; if you bundle `red` in twice, `mix` will be markedly closer to
`red`.

### Recipe 3 &mdash; round-trip through a pickle

```python
import os
from hdlib.vector import Vector

v = Vector(name="persist_me", seed=42, tags={"alice", "demo"})
path = "/tmp/persist_me.pkl"
if os.path.exists(path):
    os.remove(path)                 # dump() refuses to overwrite

v.dump(to_file=path)
reloaded = Vector(from_file=path)
assert reloaded.name == "persist_me"
assert reloaded.tags == {"alice", "demo"}
assert (reloaded.vector == v.vector).all()
```

### Recipe 4 &mdash; build a Vector from an existing numpy array

```python
import numpy as np
from hdlib.vector import Vector

raw = np.where(np.random.default_rng(0).random(10000) > 0.5, 1, -1)
v = Vector(name="from_array", vector=raw, vtype="bipolar")
print(v.size, v.vtype)              # 10000 bipolar
```

When you supply `vector=...`:
- `size` is *ignored* (taken from `len(vector)` instead).
- You are responsible for ensuring the data matches the declared `vtype`.
  The constructor does not validate this.

## Common pitfalls

- **Renaming a vector after inserting it into a `Space` corrupts the
  index.** Names are the primary key. If you must rename, remove the
  vector from the space first, reassign `.name`, and re-insert.
- **Modifying `vector.tags` directly does not update `Space.tags`.** Always
  go through `Space.add_tag(...)` / `Space.remove_tag(...)` for in-space
  tagging.
- **Sum/product of bipolar vectors is not bipolar.** After `+` or `-` the
  vector may contain values like `-3, -1, 1, 3` (for sums) or arbitrary
  integers (for differences). Call `normalize()` before using the vector
  as a clean MAP element.
- **Bipolar vs binary cannot mix.** All operators and `dist` require
  identical `vtype`; the message is `"Vectors must be of the same type"`
  or `"Vector types are not compatible"`.
- **`Vector(from_file=...)` plus a version mismatch only prints a
  warning &mdash; it does not raise.** Code that depends on strict version
  matching must check `loaded.version` itself.

## See also

- `hdlib-arithmetic` for the underlying `bind` / `bundle` / `subtraction`
  / `permute` functions and the algebraic properties they satisfy.
- `hdlib-space` for storing many `Vector` objects together.
- `hdlib-distance` for picking the right metric.
- `hdlib-reproducibility` for using `seed` consistently across vectors,
  spaces, and models.
- `hdlib-pitfalls` for a longer catalog of common errors.
