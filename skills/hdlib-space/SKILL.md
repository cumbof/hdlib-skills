---
name: hdlib-space
description: Use when working with the hdlib.space.Space class - a hashmap-like container that holds many Vector objects sharing the same size and vtype. Covers insert / bulk_insert / remove, tag indexing (add_tag / remove_tag / get(tags=...)), parent-child links (link, set_root), iteration order, similarity search with find / find_all (with thresholds), and pickle dump / load. Load whenever the user writes Space(...), space.insert(...), space.get(...), space.find(...), space.bulk_insert(...), or anything else that touches a Space.
---

# `hdlib.space.Space` &mdash; a hashmap of hypervectors

A `Space` is the natural container for a set of `Vector` objects that all
share the same dimensionality and vector type. Internally it is an
`OrderedDict` keyed by vector name, plus a secondary index from tags to
vector names, plus an optional "root" pointer that lets vectors form a
tree-shaped graph through `parents` / `children`.

## When to use this skill

- The user writes `Space(...)`, `space.insert(...)`, `space.bulk_insert(...)`.
- The user looks up vectors by name or tag (`space.get(...)`).
- The user wants to know whether a vector is in a space (`name in space`)
  or iterate over the names (`for name in space:`).
- The user wants the closest vector to a query (`space.find(...)`,
  `space.find_all(...)`).
- The user links vectors (`space.link(...)`), sets a root, or wants to
  walk a tree of vectors.
- The user pickles a space (`space.dump(...)`, `Space(from_file=...)`).

## Import

```python
from hdlib.space import Space
# Vector is re-exported from the same module for convenience:
from hdlib.space import Space, Vector
```

## Constructor

```python
Space(
    size: int = 10000,
    vtype: str = "bipolar",       # "binary" or "bipolar"
    from_file: Optional[str] = None,
)
```

- All vectors inserted into this space **must** match `size` and `vtype`.
  Mismatches raise `Exception` at `insert(...)` time, not at construction.
- `from_file` loads a pickle dump and overwrites every attribute on `self`.
  As with `Vector`, version skew between the pickle and the running hdlib
  only prints a warning.
- The constructor itself raises `ValueError` if `vtype` is not in
  `{"binary", "bipolar"}` and `FileNotFoundError` if `from_file` is set but
  the file is missing.

## Core attributes

| Attribute | Type | Notes |
|:----------|:-----|:------|
| `space.space` | `collections.OrderedDict[str, Vector]` | The actual name -> vector map. Iteration order is insertion order. |
| `space.size`, `space.vtype` | `int`, `str` | Inherited by every vector inserted via `bulk_insert`. |
| `space.tags` | `dict[Union[str, int, float], set[str]]` | tag -> set of vector names. |
| `space.root` | `Optional[str]` | Name of the root vector (when used as a tree). |
| `space.version` | `str` | hdlib version that created the space. |

## Inserting vectors

### `insert(vector)`

Insert a single `Vector`. Pre-conditions:

- `vector.size == space.size`
- `vector.vtype == space.vtype`
- `vector.name not in space`

Violating any of those raises `Exception` with a descriptive message. After
the call, every tag on `vector` is added to `space.tags` automatically.

```python
from hdlib.space import Space, Vector

space = Space(size=10000, vtype="bipolar")
space.insert(Vector(name="apple", tags={"fruit"}))
print("apple" in space)         # True
print(space.tags["fruit"])      # {"apple"}
```

### `bulk_insert(names, tags=None, ignore_existing=False)`

Generate fresh random vectors (size and vtype inherited from the space) and
insert them in one call. `names` must be a list of primitives;
`tags` (optional) must be a list with the same length where each item is a
list of tags for the corresponding name.

```python
space.bulk_insert(
    names=["banana", "cherry"],
    tags=[["fruit", "yellow"], ["fruit", "red"]],
)
```

If a name is already in the space, `bulk_insert` raises `Exception` unless
`ignore_existing=True`, in which case it silently skips the duplicate.

> **Gotcha:** `bulk_insert` calls `set(names)` internally, so duplicates in
> the input list are deduplicated *before* iteration. The order in which the
> resulting unique names are inserted is determined by Python's set order
> (effectively unspecified).

## Lookups

### `space.get(names=..., tags=...)`

Exactly one of `names` or `tags` must be provided. Returns a `list[Vector]`.

```python
fruits = space.get(tags=["fruit"])          # list of Vectors with that tag
apple  = space.get(names=["apple"])[0]      # always returns a list, even for 1
```

- `names` entries are converted to `str` (so passing `int`/`float` works).
- `tags` entries must be primitives. Multiple tags in the list are OR'd:
  vectors that have *any* of the listed tags are returned (each appears
  only once thanks to the internal `set`).
- Passing both `names` and `tags` raises `Exception` &mdash; the call refuses
  ambiguous queries.

### `name in space`, `len(space)`

```python
"apple" in space                  # True / False
len(space)                        # number of vectors in the space
```

### Iteration

`Space.__iter__` yields **names** (strings) in insertion order:

```python
for name in space:
    print(name, space.get(names=[name])[0].vtype)
```

The iterator has an internal cursor; iterating to exhaustion resets it,
so re-iterating in the same loop is safe.

### `space.memory()`

Convenience for `list(space.space.keys())` &mdash; returns all vector names.

## Tags

```python
space.add_tag("apple", "snack")           # adds to both vector.tags and space.tags["snack"]
space.remove_tag("apple", "fruit")        # remove a single tag
```

- Both methods raise `TypeError` if name or tag is not a primitive.
- Both raise `Exception` if the vector is not in the space.
- `remove_tag` is a no-op if the tag is not currently associated.
- When the last vector with a given tag is untagged or removed, the tag
  itself is dropped from `space.tags`.

> **Always use `add_tag` / `remove_tag`** instead of mutating
> `vector.tags` directly. Direct mutation leaves `space.tags` stale and
> breaks `space.get(tags=...)`.

## Removing vectors

```python
removed = space.remove("apple")    # returns the Vector
```

- Raises `Exception` when the name is missing.
- Drops every tag entry that pointed at this vector.
- Does **not** remove the name from any other vector's `parents` /
  `children` set. If you maintain a graph, you must clean links yourself.

## Links between vectors (directed graph)

```python
space.link("apple", "fruit_root")   # adds "fruit_root" to apple.children
                                    # and "apple" to fruit_root.parents
space.set_root("fruit_root")        # store the name as space.root
```

- `link` raises `Exception` if either name is not in the space.
- Links are directed; call `link(a, b)` and `link(b, a)` for bidirectional.
- This is the substrate the `GraphModel` builds on; you rarely need to
  manage links manually for plain MAP arithmetic.

## Similarity search

### `space.find(query, threshold=np.inf, method="cosine")`

Returns the `(name, distance)` of the closest vector. Internally delegates
to `find_all`.

```python
import numpy as np
from hdlib.space import Space, Vector

space = Space()
space.insert(Vector(name="a", seed=1))
space.insert(Vector(name="b", seed=2))
space.insert(Vector(name="c", seed=3))

query = Vector(seed=1)              # same data as "a"
name, dist = space.find(query)
# name == "a", dist == 0.0
```

### `space.find_all(query, threshold=np.inf, method="cosine")`

Returns `(distances: dict[name, float], best_name: str)`. Only vectors whose
distance is `<= threshold` appear in the dict.

- `method` must be one of `"cosine"`, `"euclidean"`, `"hamming"`.
- `threshold` must be `>= 0.0` (otherwise raises `ValueError`).
- If `query.size != space.size`, raises `Exception`.

```python
distances, best = space.find_all(query, method="cosine", threshold=0.5)
# distances includes only space members within cosine distance 0.5 of query
```

## Persistence

```python
space.dump(to_file="space.pkl")         # default: ./space.pkl
loaded = Space(from_file="space.pkl")
```

Identical semantics to `Vector.dump` / `Vector(from_file=...)`:

- `dump` writes to `space.pkl` in the current directory if `to_file` is omitted.
- `dump` raises `Exception` if the destination already exists (no overwrite).
- Loading a pickle from a different hdlib version prints a warning.
- The entire dictionary (including every contained `Vector` object) is
  serialised in one shot &mdash; do **not** mix `Space` pickles produced by
  different hdlib major versions.

## Full recipe &mdash; build a Space from names plus tags and search it

```python
import numpy as np
from hdlib.space import Space, Vector
from hdlib.arithmetic import bundle

space = Space(size=10000, vtype="bipolar")
space.bulk_insert(
    names=["apple", "banana", "cherry", "celery", "carrot"],
    tags=[
        ["fruit", "red"],
        ["fruit", "yellow"],
        ["fruit", "red"],
        ["vegetable", "green"],
        ["vegetable", "orange"],
    ],
)

# Build a composite "red things" vector from every red item
red_items = space.get(tags=["red"])
red_composite = red_items[0]
for v in red_items[1:]:
    red_composite = bundle(red_composite, v)
red_composite.name = "red_composite"
space.insert(red_composite)

# Which existing vector is closest to the composite?
best, dist = space.find(red_composite)
print(best, dist)   # "red_composite" itself (dist ~ 0); next-closest -> apple/cherry
```

## Common pitfalls

- **Setting `vector.tags` directly** &mdash; bypasses `space.tags` and the
  tag-based search will silently miss the vector. Use `space.add_tag(...)`.
- **Inserting a vector with the wrong `size` or `vtype`** &mdash; raises
  `Exception` at `insert`. Build vectors via `space.bulk_insert` to inherit
  these automatically, or pass `size=space.size, vtype=space.vtype` when
  constructing a `Vector` manually.
- **Renaming a vector already in the space** &mdash; the space index still
  points at the old name. Always `remove(...)` -> rename -> `insert(...)`.
- **`space.find` on an empty space** &mdash; returns `(None, np.inf)`. Treat
  the empty case explicitly.
- **`Space(size=...)` accepts any int, including tiny ones.** Pre-2.0
  versions enforced `size >= 1000`; current code does not. Choose
  `size >= 1000` yourself for HDC semantics.
- **`bulk_insert` deduplicates `names` silently** via `set(names)`. Pass
  each name only once and do not depend on input order.

## See also

- `hdlib-vectors` for the `Vector` class that fills the space.
- `hdlib-arithmetic` for how the vectors stored in a space are combined.
- `hdlib-distance` for choosing `method=` on `find` / `find_all`.
- `hdlib-graph` for the `GraphModel`, which builds on top of `Space` links.
- `hdlib-reproducibility` for keeping `bulk_insert` deterministic
  (hint: it currently uses an internal RNG without a seed; if reproducibility
  is critical, build vectors with `Vector(seed=...)` and `insert` them
  individually).
