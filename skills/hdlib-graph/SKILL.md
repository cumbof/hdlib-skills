---
name: hdlib-graph
description: Use when encoding a directed or undirected weighted graph into hyperdimensional vectors with hdlib.model.graph.GraphModel for edge prediction, error rate evaluation, error mitigation, or class labelling of edge sets. Covers constructor (size, directed, seed), fit (build node/weight/graph memories), edge_exists (query whether a specific edge is present), error_rate / error_mitigation, predict (assign a weight class to a set of edges), and the private reserved names GRAPH_ID / WEIGHT_ID.
---

# `GraphModel` &mdash; weighted graphs in HD space

`hdlib.model.graph.GraphModel` represents weighted (multi-)graphs as a
single hyperdimensional vector, following
[Poduval et al. 2022](https://doi.org/10.3389/fnins.2022.757125). Each node,
each edge weight, and the overall graph all live as `Vector`s in one
`Space`. Edge queries are answered by binding/unbinding and measuring
cosine distance.

## When to use this skill

- The user wants to encode a graph (with optional edge weights) in HD.
- The user calls `GraphModel(...)`, `.fit(edges)`, `.edge_exists(...)`,
  `.error_rate()`, `.error_mitigation(...)`, or `.predict(...)`.
- The user is implementing a pangenome / de-Bruijn-graph / knowledge-graph
  application.

## Import

```python
from hdlib.model import GraphModel
# or
from hdlib.model.graph import GraphModel
```

## Constructor

```python
GraphModel(
    size: int = 10000,
    directed: bool = False,
    seed: Optional[int] = None,
)
```

- `size` must be an `int` (else `TypeError`).
- `seed` must be `int | None` (else `TypeError`).
- The internal vector type is **always `"bipolar"`** &mdash; binary is not
  supported.
- The constructor immediately creates an empty `Space`:

  ```python
  self.space = Space(size=self.size, vtype="bipolar")
  ```

## Internal reserved names

Two special names are reserved inside the space:

| Constant | Value (string) | Role |
|:---------|:---------------|:-----|
| `GRAPH_ID` | `"__graph__"` | The single vector that represents the whole graph |
| `WEIGHT_ID` | `"__weight__"` | Prefix for vectors representing each unique edge weight |

You **must not** use these strings or strings starting with `"__weight__"`
as node names; `_add_edge` raises `ValueError` if you do.

## Edges in / out

Edges are 3-tuples `(source, target, weight)`:

- `source: str` &mdash; node name (will be cast to string).
- `target: str` &mdash; node name.
- `weight: Any` &mdash; the edge weight or class label. Anything hashable
  works (int, float, str). One weight per edge.

For multigraphs (multiple weights between the same pair of nodes),
re-call `_add_edge` (which `fit` does for each entry in `edges`) with the
same nodes and different weights.

## `fit(edges, build_nodes_memory=True)`

```python
edges = {
    ("a", "b", 1),
    ("b", "c", 2),
    ("c", "a", 1),
}
graph = GraphModel(size=10000, directed=False, seed=0)
graph.fit(edges)
```

Pre-condition: `edges` must be non-empty (else `ValueError`).

`fit` does:

1. For each `(node1, node2, weight)`:
   - If `node1` or `node2` is missing from the space, create a random
     bipolar `Vector` for it with extra attributes:
     - `memory: Vector | None` &mdash; will hold this node's neighbour bundle.
     - `weights: dict[str, set]` &mdash; per-neighbour set of weights.
   - Insert a directed link from `node1` to `node2` (and the reverse if
     `not self.directed`).
   - Store the weight in `node1.weights[node2]` (and symmetrically).
2. Build a vector per distinct edge weight:
   `Vector(name=f"__weight__{w}")` (only when `build_nodes_memory=True`).
3. For each node, compute its **memory** vector as the bundle of
   `(weight_vector * neighbour_vector)` across every outgoing neighbour.
4. Build the **graph** vector as the bundle across every node of
   `node_vector * permute(node_memory, 1)` (directed) or
   `node_vector * node_memory` (undirected). For undirected graphs the
   final graph vector is divided by 2 to undo the double-counting of
   each edge.
5. Store the graph vector in the space under `GRAPH_ID`.

Setting `build_nodes_memory=False` skips steps 2-3 &mdash; useful when
`fit` is being invoked repeatedly during error mitigation and the per-node
memories should not be re-built.

After `fit`, the space contains:

- One vector per **unique node** (with `parents`/`children`/`memory`/`weights` attributes).
- One vector per **unique weight** (prefixed with `"__weight__"`).
- One graph vector named `"__graph__"`.

The set of original edges is preserved in `self.edges`.

## `edge_exists(node1, node2, weight, threshold=None)`

```python
exists, distance, threshold_used = graph.edge_exists("a", "b", weight=1)
```

Returns:
- `exists: bool` &mdash; True if `distance < threshold_used`.
- `distance: float` &mdash; cosine distance between
  `weight_vector * node2_vector` and the unbound `node1_memory`.
- `threshold_used: float` &mdash; the threshold actually applied (the
  argument if non-None, else an automatically estimated node-specific
  threshold).

### Auto threshold

When `threshold` is `None`, the function:

1. Selects up to 10 of `node1.children` (real neighbours).
2. Samples the same number of random non-neighbours from the entire
   space (using `self.rand`).
3. Computes cosine distances of all of them against the unbound memory.
4. Uses the **5th percentile** of those distances as the threshold.

The threshold is therefore *per node and per weight*. The graph caches
it in `self.weight_to_node_specific_thresholds` so subsequent queries
on the same `node1 / weight` reuse it.

### Errors

- `Exception("There is no graph in space")` &mdash; `fit` was not called.
- `Exception("Node '...' not in space")` &mdash; unknown source or target.
- `Exception("Weight vector '...' not in space")` &mdash; the requested
  weight was never inserted via `fit`.

## `error_rate()`

```python
fnr, false_negatives = graph.error_rate()
```

Returns:
- `fnr: float` &mdash; false-negative rate (fraction of `self.edges` whose
  `edge_exists` returned `False`).
- `false_negatives: set[tuple]` &mdash; the actual missed edges.

The function populates the threshold cache as a side effect.

## `error_mitigation(max_iter=10, nproc=1)`

Iteratively re-binds misclassified edges to reduce the false-negative
rate:

```python
graph.error_mitigation(max_iter=10, nproc=4)
```

Loop (up to `max_iter` times):

1. Estimate the current `error_rate` (optionally in parallel across edge
   chunks via `multiprocessing.Pool` when `nproc > 1`).
2. For each false-negative edge `(n1, n2, w)`, augment the graph vector
   with a "reinforcement signal":
   `graph_vector += node1 * permute(weight * node2, 1)` (directed)
   or without the permutation for undirected.
3. Recompute the error rate. If it got worse, stop and discard this
   iteration's update.

Prints per-iteration diagnostics to stdout. The graph vector is updated
in place when an iteration improved the error rate.

## `predict(edges)`

Classifies a set of test edges by majority vote across weights:

```python
predicted_weight, percentage = graph.predict({
    ("a", "b", any_weight),
    ("b", "c", any_weight),
})
```

For each test edge `(n1, n2, ignored_weight)`:
1. Call `edge_exists(n1, n2, w)` for every `w in self.weights`.
2. Vote for the weight(s) with the smallest distance (using a relative
   tolerance internally).

Returns the most-voted weight and the percentage of edges that
contributed votes to it. Note: the `weight` in each test tuple is
ignored &mdash; the method *predicts* the weight.

The static `_predict(instance, name, y_true, edges)` is a multiprocessing
wrapper returning `(name, y_true, predict(edges))`.

## Worked example &mdash; classify a graph by its edges

```python
from hdlib.model import GraphModel

# Build a model from labelled edges
train_edges = {
    ("a", "b", "class_red"),
    ("b", "c", "class_red"),
    ("a", "c", "class_red"),
    ("d", "e", "class_blue"),
    ("e", "f", "class_blue"),
    ("d", "f", "class_blue"),
}
graph = GraphModel(size=10000, directed=False, seed=0)
graph.fit(train_edges)

# Check whether a known edge is detected
exists, dist, thr = graph.edge_exists("a", "b", "class_red")
print(exists, dist, thr)

# Estimate / mitigate error
fnr, missed = graph.error_rate()
print(f"FNR before mitigation: {fnr:.4f}")
graph.error_mitigation(max_iter=5)
print(f"FNR after mitigation:  {graph.error_rate()[0]:.4f}")

# Classify an unseen set of edges
prediction, pct = graph.predict({("x", "y", None), ("y", "z", None)})
print(f"unseen edges -> {prediction} with {pct:.1f}% support")
```

## Common pitfalls

- **Using a node name starting with `"__weight__"` or equal to
  `"__graph__"`** raises `ValueError`. Use any other naming scheme.
- **`edge_exists` on a model that was never `fit`** raises `Exception("There is no graph in space")`.
- **Repeating identical edges in `self.edges`** does not increase the
  weight of that connection in the graph vector beyond the first
  insertion; `_add_edge` checks for existence before adding.
- **Different `weight` for the same `(node1, node2)`** is permitted; both
  weights live in `node1.weights[node2]` and produce different bindings
  in the node memory.
- **Error mitigation modifies the graph vector in place**, so subsequent
  `edge_exists` calls see the updated representation. If you need the
  original, snapshot before calling `error_mitigation`.
- **`error_rate` ignores false positives.** It only measures missed
  edges (false negatives). Build your own evaluation if you care about
  unintended edges showing up as "exists".
- **The auto threshold is randomised.** `edge_exists(..., threshold=None)`
  draws random non-neighbours via `self.rand`. For deterministic
  thresholds, either pass `threshold=...` explicitly or always seed the
  model via `GraphModel(seed=...)`.

## See also

- `hdlib-vectors`, `hdlib-arithmetic`, `hdlib-space` &mdash; the underlying
  primitives.
- `hdlib-pitfalls` &mdash; for shared issues across hdlib models.
- `examples/pangenome` in the hdlib repo &mdash; a complete graph-based
  application (de Bruijn graphs of viral pangenomes).
