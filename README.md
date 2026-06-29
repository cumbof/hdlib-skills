# hdlib-skills

A collection of [Agent Skills](https://agentskills.io) that teach LLM agents
how to use [`hdlib`](https://github.com/cumbof/hdlib) &mdash; a Python library for
building Vector-Symbolic Architectures (a.k.a. Hyperdimensional Computing).

These skills make it easy for any skill-aware coding assistant (Claude Code,
GitHub Copilot CLI, and other tools that implement the open
[Agent Skills standard](https://agentskills.io)) to write correct, idiomatic
`hdlib` code on demand &mdash; without burning context on the full library
documentation.

> Target library: [`cumbof/hdlib`](https://github.com/cumbof/hdlib) v2.1.0+

## What is a skill?

A skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```yaml
---
name: hdlib-vectors
description: Use when you need to create, combine, or compare hyperdimensional Vector objects with hdlib (binary/bipolar vectors, dist, normalize, dump). Covers hdlib.vector.Vector.
---

# Markdown body with detailed instructions and code examples
```

The `description` is what the agent sees by default; it decides when to load
the skill. The body of `SKILL.md` is only injected into the agent's context
once the skill is invoked &mdash; so detailed reference material costs nothing
until it's needed.

## Repository layout

```
hdlib-skills/
├── README.md                       # This file
├── CONTRIBUTING.md                 # How to add or modify skills
├── LICENSE                         # MIT
└── skills/                         # All skills live here
    ├── README.md                   # Index with descriptions
    ├── hdlib-overview/             # Master skill - load this first
    │   └── SKILL.md
    ├── hdlib-installation/
    ├── hdlib-vectors/
    ├── hdlib-space/
    ├── hdlib-arithmetic/
    ├── hdlib-distance/
    ├── hdlib-classification/
    ├── hdlib-feature-selection/
    ├── hdlib-hyperparameter-tuning/
    ├── hdlib-clustering/
    ├── hdlib-regression/
    ├── hdlib-graph/
    ├── hdlib-quantum-arithmetic/
    ├── hdlib-quantum-classification/
    ├── hdlib-quantum-advanced/
    ├── hdlib-analogical-reasoning/
    ├── hdlib-encoding-data/
    ├── hdlib-pitfalls/
    └── hdlib-reproducibility/
```

## Installing the skills

### Claude Code

Copy the `skills/` directory under either of:

```bash
# Personal (available across all your projects)
cp -r skills/* ~/.claude/skills/

# Project (only this project)
mkdir -p .claude/skills
cp -r skills/* .claude/skills/
```

After the copy, Claude Code picks up the skills automatically (no restart
needed for personal/project additions).

### GitHub Copilot CLI

Place the skills under your Copilot CLI skills path (consult the Copilot CLI
documentation for the exact location on your platform), or invoke them
manually by referencing the relevant `SKILL.md`.

### Any other Agent Skills-compatible tool

Follow your tool's instructions for the open Agent Skills standard. The
directory layout (`<skill-name>/SKILL.md` with YAML frontmatter) is portable.

## Skill index

See [`skills/README.md`](./skills/README.md) for the complete index with
descriptions of every skill in this repository.

### Foundation skills

| Skill | Use when... |
|:------|:------------|
| `hdlib-overview` | First touch with hdlib &mdash; gives the big picture and points to the other skills |
| `hdlib-installation` | Installing hdlib or wiring up its dependencies (numpy, scikit-learn, qiskit) |
| `hdlib-vectors` | Working with the `Vector` class (creation, operators, dist, normalize, dump) |
| `hdlib-space` | Organising vectors in a `Space` (insert, tags, links, search, persistence) |
| `hdlib-arithmetic` | Using the MAP operators: `bundle`, `bind`, `subtraction`, `permute` |
| `hdlib-distance` | Choosing between cosine, Euclidean, and Hamming distance for HD vectors |

### Model skills

| Skill | Use when... |
|:------|:------------|
| `hdlib-classification` | Supervised classification with `ClassificationModel` |
| `hdlib-feature-selection` | HDC-based stepwise feature selection (forward/backward) |
| `hdlib-hyperparameter-tuning` | Parameter sweep on vector size and number of levels |
| `hdlib-clustering` | Unsupervised clustering with `ClusteringModel` (HDC k-means) |
| `hdlib-regression` | RegHD multi-model regression (`RegressionEncoder`, `RegressionModel`) |
| `hdlib-graph` | Encoding directed/undirected weighted graphs with `GraphModel` |

### Quantum skills

| Skill | Use when... |
|:------|:------------|
| `hdlib-quantum-arithmetic` | Quantum MAP arithmetic (encode, bundle, bind, permute) and oracle compression |
| `hdlib-quantum-classification` | `QuantumClassificationModel` on simulators or IBM Quantum backends |
| `hdlib-quantum-advanced` | Compute-uncompute test, `superposition_bundle`, `entangled_bind`, Grover search |

### Pattern skills

| Skill | Use when... |
|:------|:------------|
| `hdlib-analogical-reasoning` | Implementing role-filler analogies ("Dollar of Mexico"-style queries) |
| `hdlib-encoding-data` | Choosing how to encode tabular, sequence, or graph data into hypervectors |
| `hdlib-pitfalls` | Debugging unexpected results, type/size mismatches, normalisation issues |
| `hdlib-reproducibility` | Making hdlib runs deterministic with seeds |

## How the skills were authored

Each skill was written against the actual source code of `cumbof/hdlib` and the
docstrings within it. Code blocks marked as runnable were tested against the
public API. See `CONTRIBUTING.md` for how to add or update a skill.

## License

MIT &mdash; see [`LICENSE`](./LICENSE).
