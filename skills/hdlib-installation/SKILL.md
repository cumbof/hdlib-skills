---
name: hdlib-installation
description: Use when you need to install, uninstall, or upgrade the hdlib Python library, set up a virtual environment for it, or diagnose dependency / version errors involving numpy, scikit-learn, qiskit, qiskit-aer, qiskit-ibm-runtime, qiskit_machine_learning, mthree, scipy, or tabulate. Covers pip, conda, and editable installs from source.
---

# Installing `hdlib`

`hdlib` is distributed both on **PyPI** and on **conda-forge**. Pick whichever
package manager the user is already using. The package name in both
ecosystems is exactly `hdlib`.

## Python and OS support

- Python **>= 3.11** (the `setup.py` lists 3.11, 3.12 and 3.13 in its
  classifiers; the `python_requires` constraint is `">=3.11"`).
- Pure-Python; works on Linux, macOS, and Windows.
- The quantum stack relies on Qiskit which builds on every officially
  supported Python version above.

## Installing the released version

```bash
# pip (recommended on most platforms)
python -m pip install --upgrade hdlib

# conda-forge
conda install -c conda-forge hdlib
```

Always pin the version when authoring a reproducible script:

```bash
python -m pip install 'hdlib==2.1.0'
```

## Verifying the install

```python
import hdlib
print(hdlib.__version__)        # -> "2.1.0"

from hdlib.vector import Vector
from hdlib.space import Space
from hdlib.arithmetic import bind, bundle, permute, subtraction
from hdlib.model import ClassificationModel, GraphModel, ClusteringModel
from hdlib.model.regression import RegressionEncoder, RegressionModel
```

All of these imports should succeed against a clean install. If any of them
raise `ImportError` you're either on an older `hdlib` (before 2.0 the API
was different &mdash; see CHANGELOG) or a dependency is missing.

## Dependencies

`hdlib` declares the following runtime dependencies (`requirements.txt`):

```
mthree>=3.0.0
numpy>=2.3.4
qiskit>=2.2.1
qiskit-aer>=0.17.2
qiskit-ibm-runtime>=0.42.0
qiskit_machine_learning>=0.8.4
scikit-learn>=1.7.2
scipy>=1.16.2
tabulate>=0.9.0
```

What each one is used for inside hdlib:

| Package | Used in |
|:--------|:--------|
| `numpy` | Vector storage, all classical arithmetic, distance metrics |
| `scikit-learn` | `StratifiedKFold` for `ClassificationModel.cross_val_predict`, metric scorers |
| `scipy` | `softmax` in `RegressionModel` |
| `qiskit` | Quantum circuit construction in `hdlib.arithmetic.quantum` and `QuantumClassificationModel` |
| `qiskit-aer` | Local simulator used by the quantum models |
| `qiskit-ibm-runtime` | Optional submission to IBM Quantum hardware |
| `qiskit_machine_learning` | Imported by example scripts &mdash; not by the library itself, but the dependency is declared |
| `mthree` | Readout-error mitigation used by `run_compute_uncompute_test` on hardware |
| `tabulate` | Used by the example `chopin2.py` script that ships with the repo |

If the user only intends to use classical HDC (no quantum models, no IBM
runtime), the qiskit-stack still gets installed because the library imports
qiskit at module import time inside `hdlib.model.classification` (for
`QuantumClassificationModel`). Skipping the quantum dependencies is **not**
supported &mdash; do not advise the user to install hdlib without them.

## Installing from source / editable mode

For local development against the latest `main`:

```bash
git clone https://github.com/cumbof/hdlib.git
cd hdlib
python -m pip install -e ".[dev]"   # installs runtime + dev extras (build, twine, pytest)
```

The `dev` extra is declared in `setup.py` &mdash; it brings in `pytest` (>=8)
and packaging tools. To get just the tests:

```bash
python -m pip install -e ".[test]"
```

Run the bundled test suite from the repo root:

```bash
pytest -q
```

(The suite lives in `test/test.py` and is configured via `pyproject.toml`.)

## Virtual environment recipe (recommended)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'hdlib==2.1.0'
```

Confirm:

```bash
python -c "import hdlib, qiskit, sklearn; print(hdlib.__version__)"
```

## IBM Quantum access (optional)

The `QuantumClassificationModel` and the `run_compute_uncompute_test`
function can submit jobs to real IBM Quantum hardware. To enable that, the
user needs:

1. An IBM Quantum account and API token (free tier available).
2. `qiskit-ibm-runtime` (already a hdlib dependency).
3. To pass `channel`, `instance`, `backend`, and `api_key` to the quantum
   model, *or* to pre-save credentials via:

   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                     token="...",
                                     overwrite=True)
   ```

When no credentials are provided, the model falls back to `AerSimulator`
locally (with optional GPU device acceleration if Qiskit Aer was built with
CUDA support).

## Common installation errors

- `ImportError: cannot import name '__version__' from 'hdlib'` &mdash; The
  user is on a pre-2.0 hdlib install (e.g. 0.1.x). Tell them to
  `pip install --upgrade hdlib`.
- `ModuleNotFoundError: No module named 'qiskit'` &mdash; The qiskit pin failed
  during install. Suggest `pip install --upgrade 'qiskit>=2.2.1'`.
- `numpy.AxisError: axis ... is out of bounds` or `AttributeError: module
  'numpy' has no attribute 'Inf'` &mdash; The user is on an old numpy. Upgrade
  to numpy >=2.3.4 (see `requirements.txt`; older numpy 1.x will not work).
- `ImportError: cannot import name 'AerSimulator' from 'qiskit_aer'` &mdash;
  Their `qiskit-aer` is too old. `pip install --upgrade qiskit-aer`.

## Uninstalling

```bash
python -m pip uninstall hdlib
```

This does not remove the qiskit / scikit-learn dependencies; remove them
manually if no other package needs them.

## See also

- `hdlib-overview` for the high-level architecture of the library.
- `hdlib-pitfalls` for runtime errors caused by version skew between
  hdlib-generated pickle files and the running hdlib version.
