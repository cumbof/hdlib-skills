# Skills index

This directory contains all agent skills targeting [`cumbof/hdlib`](https://github.com/cumbof/hdlib).
Every skill lives in its own directory with a `SKILL.md` file that includes
YAML frontmatter declaring its `name` and `description`.

Browse by category below, or start with `hdlib-overview` if you don't know
which one you need.

## Foundation skills

These cover the building blocks of every hdlib program.

| Directory | Description |
|:----------|:------------|
| [`hdlib-overview`](./hdlib-overview/) | Master skill &mdash; high-level architecture of hdlib and pointers to the other skills. Load first when the user mentions hdlib for the first time. |
| [`hdlib-installation`](./hdlib-installation/) | Installation, Python/numpy version requirements, optional quantum dependencies, and pip/conda commands. |
| [`hdlib-vectors`](./hdlib-vectors/) | The `hdlib.vector.Vector` class: creation, binary/bipolar types, operators (`+`, `-`, `*`), `dist`, `normalize`, pickle dump/load. |
| [`hdlib-space`](./hdlib-space/) | The `hdlib.space.Space` class: inserting/removing vectors, tags, links, `find`/`find_all`, iteration, pickle dump/load. |
| [`hdlib-arithmetic`](./hdlib-arithmetic/) | The MAP operators in `hdlib.arithmetic`: `bundle`, `bind`, `subtraction`, `permute`. Properties and edge cases. |
| [`hdlib-distance`](./hdlib-distance/) | Choosing between `cosine`, `euclidean`, and `hamming` distance for hyperdimensional vectors and setting thresholds. |

## Model skills

Building blocks for higher-level VSA/HDC models.

| Directory | Description |
|:----------|:------------|
| [`hdlib-classification`](./hdlib-classification/) | `hdlib.model.classification.ClassificationModel`: fit/predict/retrain workflow, level vectors, cross-validation. |
| [`hdlib-feature-selection`](./hdlib-feature-selection/) | `ClassificationModel.stepwise_regression` for backward variable elimination and forward variable selection. |
| [`hdlib-hyperparameter-tuning`](./hdlib-hyperparameter-tuning/) | `ClassificationModel.auto_tune` parameter sweep on vector size and number of level vectors. |
| [`hdlib-clustering`](./hdlib-clustering/) | `hdlib.model.clustering.ClusteringModel` &mdash; HDC-based k-means clustering. |
| [`hdlib-regression`](./hdlib-regression/) | `hdlib.model.regression.RegressionEncoder` and `RegressionModel` &mdash; RegHD multi-model regression. |
| [`hdlib-graph`](./hdlib-graph/) | `hdlib.model.graph.GraphModel` for directed/undirected weighted graphs with edge prediction and error mitigation. |

## Quantum skills

For Quantum Hyperdimensional Computing.

| Directory | Description |
|:----------|:------------|
| [`hdlib-quantum-arithmetic`](./hdlib-quantum-arithmetic/) | Quantum versions of encode/bundle/bind/permute, plus `compress_circuit` and `statevector_to_bipolar`. |
| [`hdlib-quantum-classification`](./hdlib-quantum-classification/) | `QuantumClassificationModel`: simulator vs IBM Quantum backends, noise models, retrain. |
| [`hdlib-quantum-advanced`](./hdlib-quantum-advanced/) | Compute-uncompute similarity, `superposition_bundle`, `entangled_bind`, `grover_search`, `quantum_majority_bundle`, `quantum_contextual_bind`, circuit metrics. |

## Pattern skills

Higher-level usage patterns and pitfalls.

| Directory | Description |
|:----------|:------------|
| [`hdlib-analogical-reasoning`](./hdlib-analogical-reasoning/) | Role-filler binding patterns (the "Dollar of Mexico" reasoning task). |
| [`hdlib-encoding-data`](./hdlib-encoding-data/) | How to encode tabular, sequence, and graph data into hypervectors so models can consume them. |
| [`hdlib-pitfalls`](./hdlib-pitfalls/) | Catalog of common mistakes (type/size mismatches, missing normalisation, file collisions on dump). |
| [`hdlib-reproducibility`](./hdlib-reproducibility/) | Seeds for reproducible runs across `Vector`, `Space`, model fit, and quantum models. |

## Authoring guidelines

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the skill-authoring style
guide. Every code block in a skill must run against the current `hdlib` main.
