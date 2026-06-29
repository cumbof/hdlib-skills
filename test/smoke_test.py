"""Smoke test for the hdlib-skills repository.

Each block exercises a canonical code snippet from one of the skills. If
this script runs end-to-end without raising and prints the expected
"PASS" markers, the skill examples are aligned with the running hdlib
version (2.1.0+).
"""

import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# hdlib-overview
# ---------------------------------------------------------------------------
def test_overview():
    from hdlib.space import Space
    from hdlib.vector import Vector
    from hdlib.arithmetic import bind, bundle

    space = Space(size=10000, vtype="bipolar")
    space.bulk_insert(names=["alice", "bob", "carol"])
    alice = space.get(names=["alice"])[0]
    bob = space.get(names=["bob"])[0]
    carol = space.get(names=["carol"])[0]

    group = bundle(bundle(alice, bob), carol)
    group.name = "group"
    space.insert(group)
    best, dist = space.find(group)
    assert best == "group" and dist < 1e-9, (best, dist)
    print("PASS hdlib-overview")


# ---------------------------------------------------------------------------
# hdlib-vectors
# ---------------------------------------------------------------------------
def test_vectors():
    from hdlib.vector import Vector

    v1 = Vector(name="v1", seed=1)
    v2 = Vector(name="v2", seed=2)
    assert 0.5 < v1.dist(v2) < 1.5, v1.dist(v2)
    assert 4000 <= v1.dist(v2, method="hamming") <= 6000

    red = Vector(name="red", seed=10)
    green = Vector(name="green", seed=20)
    blue = Vector(name="blue", seed=30)
    mix = red + green + blue
    mix.normalize()
    assert set(np.unique(mix.vector)).issubset({-1, 1})

    # Pickle round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vector.pkl")
        v_save = Vector(name="persist_me", seed=42, tags={"alice", "demo"})
        v_save.dump(to_file=path)
        v_loaded = Vector(from_file=path)
        assert v_loaded.tags == {"alice", "demo"}
        assert (v_loaded.vector == v_save.vector).all()

    # From-array constructor
    raw = np.where(np.random.default_rng(0).random(10000) > 0.5, 1, -1)
    v = Vector(name="from_array", vector=raw, vtype="bipolar")
    assert v.size == 10000 and v.vtype == "bipolar"
    print("PASS hdlib-vectors")


# ---------------------------------------------------------------------------
# hdlib-space
# ---------------------------------------------------------------------------
def test_space():
    from hdlib.space import Space, Vector
    from hdlib.arithmetic import bundle

    space = Space(size=10000, vtype="bipolar")
    # NOTE: do not use the `tags=` argument to bulk_insert - see hdlib-space pitfalls
    space.bulk_insert(names=["apple", "banana", "cherry", "celery", "carrot"])
    for name, tags in [
        ("apple",  ["fruit", "red"]),
        ("banana", ["fruit", "yellow"]),
        ("cherry", ["fruit", "red"]),
        ("celery", ["vegetable", "green"]),
        ("carrot", ["vegetable", "orange"]),
    ]:
        for t in tags:
            space.add_tag(name, t)

    red_items = space.get(tags=["red"])
    assert {v.name for v in red_items} == {"apple", "cherry"}, {v.name for v in red_items}

    composite = red_items[0]
    for v in red_items[1:]:
        composite = bundle(composite, v)
    composite.name = "red_composite"
    space.insert(composite)
    best, _ = space.find(composite)
    assert best == "red_composite"

    # Persistence
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "space.pkl")
        space.dump(to_file=path)
        reloaded = Space(from_file=path)
        assert "red_composite" in reloaded
    print("PASS hdlib-space")


# ---------------------------------------------------------------------------
# hdlib-arithmetic
# ---------------------------------------------------------------------------
def test_arithmetic():
    from hdlib.vector import Vector
    from hdlib.arithmetic import bind, bundle, permute, subtraction

    a = Vector(name="a", seed=1)
    b = Vector(name="b", seed=2)

    bound = bind(a, b)
    # bind is invertible
    recovered = bind(bound, b)
    assert (recovered.vector == a.vector).all()

    bundled = bundle(a, b)
    # bundle is similar to inputs
    assert bundled.dist(a) < bundled.dist(Vector(seed=3))

    rolled = permute(a, rotate_by=2)
    assert np.array_equal(rolled.vector, np.roll(a.vector, 2))

    diff = subtraction(a, b)
    assert diff.size == a.size

    # In-place methods
    c = Vector(name="c", seed=5)
    c_copy = Vector(vector=c.vector.copy(), name="c_copy", vtype=c.vtype)
    c.bind(b)
    assert not np.array_equal(c.vector, c_copy.vector)
    print("PASS hdlib-arithmetic")


# ---------------------------------------------------------------------------
# hdlib-distance
# ---------------------------------------------------------------------------
def test_distance():
    from hdlib.vector import Vector

    D = 10000
    v1 = Vector(seed=1, size=D)
    v2 = Vector(seed=2, size=D)

    assert v1.dist(v1) < 1e-9
    assert 0.95 < v1.dist(v2) < 1.05

    v3 = Vector(vector=-v1.vector.copy(), size=D, vtype="bipolar")
    assert abs(v1.dist(v3) - 2.0) < 1e-9

    for method in ("cosine", "euclidean", "hamming"):
        d = v1.dist(v2, method=method)
        assert d > 0
    print("PASS hdlib-distance")


# ---------------------------------------------------------------------------
# hdlib-classification (using iris)
# ---------------------------------------------------------------------------
def test_classification():
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from hdlib.model import ClassificationModel

    iris = load_iris()
    points = iris.data.tolist()
    labels = iris.target.tolist()

    model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
    model.fit(points, labels=labels, seed=42)

    # Held-out test indices
    test_indices = list(range(0, 150, 10))  # 15 samples evenly across classes
    _, preds, _, _, err, _ = model.predict(test_indices, retrain=5)
    y_true = [labels[i] for i in test_indices]
    acc = accuracy_score(y_true, preds)
    assert acc > 0.7, f"unexpectedly low accuracy {acc}"
    print(f"PASS hdlib-classification (accuracy={acc:.3f})")


# ---------------------------------------------------------------------------
# hdlib-feature-selection (using iris, small)
# ---------------------------------------------------------------------------
def test_feature_selection():
    from sklearn.datasets import load_iris
    from hdlib.model import ClassificationModel

    iris = load_iris()
    points = iris.data.tolist()
    labels = iris.target.tolist()
    features = list(iris.feature_names)

    model = ClassificationModel(size=10000, levels=10, vtype="bipolar")
    model.fit(points, labels=labels, seed=0)

    importances, scores, best_importance, count_models = model.stepwise_regression(
        points=points,
        features=features,
        labels=labels,
        method="backward",
        cv=3,
        metric="accuracy",
        threshold=0.5,
        uncertainty=10.0,
        n_jobs=1,
    )
    assert set(importances) == set(features)
    assert count_models > 0
    print(f"PASS hdlib-feature-selection (count_models={count_models})")


# ---------------------------------------------------------------------------
# hdlib-hyperparameter-tuning (very small grid)
# ---------------------------------------------------------------------------
def test_hyperparameter_tuning():
    from sklearn.datasets import load_iris
    from hdlib.model import ClassificationModel

    iris = load_iris()
    points = iris.data.tolist()
    labels = iris.target.tolist()

    model = ClassificationModel()
    best_size, best_levels, best_acc = model.auto_tune(
        points=points,
        labels=labels,
        size_range=range(2000, 4001, 2000),
        levels_range=range(5, 11, 5),
        cv=3,
        metric="accuracy",
        n_jobs=1,
    )
    # All candidates have size > len(points)=150 so we should get a winner
    assert best_size in (2000, 4000)
    assert best_levels in (5, 10)
    assert 0.0 < best_acc <= 1.0
    print(f"PASS hdlib-hyperparameter-tuning (best size={best_size}, levels={best_levels}, acc={best_acc:.3f})")


# ---------------------------------------------------------------------------
# hdlib-clustering
# ---------------------------------------------------------------------------
def test_clustering():
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    from hdlib.model import ClusteringModel

    X, y = make_blobs(n_samples=200, centers=4, n_features=8, random_state=0)
    model = ClusteringModel(k=4, n_features=X.shape[1], size=5000, seed=0)
    model.fit(X)
    pred = model.predict(X)
    ari = adjusted_rand_score(y, pred)
    assert ari > 0.5, f"unexpectedly low ARI: {ari}"
    print(f"PASS hdlib-clustering (ARI={ari:.3f})")


# ---------------------------------------------------------------------------
# hdlib-regression (very small for speed)
# ---------------------------------------------------------------------------
def test_regression():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from hdlib.model.regression import RegressionModel

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    np.random.seed(0)
    model = RegressionModel(
        D=2000,
        n_features=X.shape[1],
        k_models=4,
        learning_rate=0.005,
        iterations=3,
        binary_threshold=0.0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    print("PASS hdlib-regression")


# ---------------------------------------------------------------------------
# hdlib-graph
# ---------------------------------------------------------------------------
def test_graph():
    from hdlib.model import GraphModel

    edges = {
        ("a", "b", "class_red"),
        ("b", "c", "class_red"),
        ("a", "c", "class_red"),
        ("d", "e", "class_blue"),
        ("e", "f", "class_blue"),
        ("d", "f", "class_blue"),
    }
    graph = GraphModel(size=5000, directed=False, seed=0)
    graph.fit(edges)

    exists, dist, thr = graph.edge_exists("a", "b", "class_red")
    assert dist >= 0
    fnr, missed = graph.error_rate()
    assert 0.0 <= fnr <= 1.0
    print(f"PASS hdlib-graph (FNR={fnr:.3f})")


# ---------------------------------------------------------------------------
# hdlib-quantum-arithmetic
# ---------------------------------------------------------------------------
def test_quantum_arithmetic():
    from hdlib.arithmetic.quantum import (
        encode,
        bundle as q_bundle,
        bind as q_bind,
        permute as q_permute,
        compress_circuit,
        statevector_to_bipolar,
    )

    v1 = np.array([1, -1, 1, 1, -1, -1, 1, -1])
    v2 = np.array([-1, 1, 1, -1, 1, -1, -1, 1])
    v3 = np.array([1, 1, -1, -1, 1, 1, -1, -1])

    c1, c2, c3 = encode(v1), encode(v2), encode(v3)

    # Decode round-trip
    assert (statevector_to_bipolar(c1) == v1).all()

    # Bind product
    bound = q_bind([c1, c2])
    expected = v1 * v2
    assert (statevector_to_bipolar(bound) == expected).all()

    # Bundle (majority vote of v1, v2, v3) - use the default method="average"
    bundled = q_bundle([c1, c2, c3])
    decoded = statevector_to_bipolar(bundled)
    expected_majority = np.sign(v1 + v2 + v3)
    expected_majority[expected_majority == 0] = 1
    assert (decoded == expected_majority).all(), (decoded, expected_majority)

    # Permute - rotate_by maps to np.roll on the basis state index
    shifted = q_permute(c1, num_qubits=3, shift=2)
    expected_perm = np.roll(v1, 2)
    assert (statevector_to_bipolar(shifted) == expected_perm).all()

    # Compress shouldn't change semantics
    compressed = compress_circuit(bound)
    assert (statevector_to_bipolar(compressed) == expected).all()
    print("PASS hdlib-quantum-arithmetic")


# ---------------------------------------------------------------------------
# hdlib-quantum-classification (simulator, tiny)
# ---------------------------------------------------------------------------
def test_quantum_classification():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from qiskit_aer import AerSimulator
    from hdlib.model import QuantumClassificationModel

    iris = load_iris()
    X = MinMaxScaler().fit_transform(iris.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), iris.target.tolist(), test_size=0.3, random_state=0, stratify=iris.target
    )

    # Keep it small for CI speed: use only 30 training samples
    model = QuantumClassificationModel(size=16, levels=4, seed=42, shots=512)
    # Force CPU AerSimulator (see hdlib-pitfalls #13b)
    model.backend = AerSimulator()
    model.fit(X_train[:30], y_train[:30])
    preds, _ = model.predict(X_test[:6])
    assert len(preds) == 6
    print(f"PASS hdlib-quantum-classification (preds={preds})")


# ---------------------------------------------------------------------------
# hdlib-quantum-advanced (compute_uncompute and superposition_bundle)
# ---------------------------------------------------------------------------
def test_quantum_advanced():
    from qiskit_aer import AerSimulator
    from hdlib.arithmetic.quantum import (
        encode,
        run_compute_uncompute_test,
        superposition_bundle,
        statevector_to_bipolar,
        get_circuit_metrics,
        entangled_bind,
        quantum_majority_bundle,
    )

    backend = AerSimulator()
    # Vectors must be non-antipodal to distinguish themselves from each other:
    # use two random-ish vectors with about half their bits in agreement.
    v1 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    v2 = np.array([1, 1, -1, -1, 1, 1, -1, -1])

    queries = [encode(v1)]
    prototypes = [encode(v1), encode(v2)]
    sims, _ = run_compute_uncompute_test(queries, prototypes, backend, shots=2048, seed=0)
    # v1 has |<v1|v1>| == 1 ; |<v1|v2>| should be < 1 since they share only some bits
    assert sims[0][0] > sims[0][1], sims

    bundled = superposition_bundle([encode(v1), encode(v2)])
    decoded = statevector_to_bipolar(bundled)
    assert set(np.unique(decoded)).issubset({-1, 1})

    majority = quantum_majority_bundle([encode(v1), encode(v1), encode(v2)])
    decoded_m = statevector_to_bipolar(majority)
    assert set(np.unique(decoded_m)).issubset({-1, 1})

    eb = entangled_bind(encode(v1), encode(v2))
    assert eb.num_qubits == 1 + 3 + 3

    metrics = get_circuit_metrics(encode(v1), num_system_qubits=3, backend=backend, optimization_level=1)
    assert metrics["depth"] >= 0
    print("PASS hdlib-quantum-advanced")


# ---------------------------------------------------------------------------
# hdlib-analogical-reasoning
# ---------------------------------------------------------------------------
def test_analogical_reasoning():
    from hdlib.space import Space
    from hdlib.vector import Vector
    from hdlib.arithmetic import bind, bundle

    D = 10000
    space = Space(size=D, vtype="bipolar")
    for name in ["USA", "DOL", "WDC", "MEX", "PES", "MXC", "NAM", "CAP", "MON"]:
        space.insert(Vector(name=name, size=D, vtype="bipolar"))

    def v(name): return space.get(names=[name])[0]

    usa_record = bundle(
        bundle(bind(v("NAM"), v("USA")), bind(v("CAP"), v("WDC"))),
        bind(v("MON"), v("DOL")),
    )
    mex_record = bundle(
        bundle(bind(v("NAM"), v("MEX")), bind(v("CAP"), v("MXC"))),
        bind(v("MON"), v("PES")),
    )
    mapping = bind(usa_record, mex_record)
    guess = bind(v("DOL"), mapping)
    guess.normalize()

    name, dist = space.find(guess)
    assert name == "PES", f"expected PES, got {name} at distance {dist}"
    print(f"PASS hdlib-analogical-reasoning (closest={name}, dist={dist:.3f})")


# ---------------------------------------------------------------------------
# hdlib-encoding-data: Pattern D (role-filler)
# ---------------------------------------------------------------------------
def test_encoding_data():
    from hdlib.space import Space, Vector
    from hdlib.arithmetic import bind, bundle, permute

    D = 10000
    space = Space(size=D, vtype="bipolar")
    roles = ["COLOR", "SHAPE", "SIZE"]
    fillers = {"COLOR": ["red", "green", "blue"],
               "SHAPE": ["circle", "square", "triangle"],
               "SIZE": ["small", "medium", "large"]}
    for r in roles:
        space.insert(Vector(name=r, size=D, vtype="bipolar"))
        for f in fillers[r]:
            space.insert(Vector(name=f"{r}:{f}", size=D, vtype="bipolar"))

    def encode(sample):
        record = None
        for role, filler in sample.items():
            b = bind(space.get(names=[role])[0],
                     space.get(names=[f"{role}:{filler}"])[0])
            record = b if record is None else bundle(record, b)
        record.normalize()
        return record

    sample = {"COLOR": "red", "SHAPE": "circle", "SIZE": "large"}
    hv = encode(sample)
    assert set(np.unique(hv.vector)).issubset({-1, 1})

    # Sequence encoding (Pattern E)
    tokens = ["A", "C", "G", "T", "A", "C"]
    vocab = {}
    for t in set(tokens):
        space.insert(Vector(name=f"tok:{t}", size=D, vtype="bipolar"))
        vocab[t] = space.get(names=[f"tok:{t}"])[0]
    seq = None
    for i, tok in enumerate(tokens):
        rolled = permute(vocab[tok], rotate_by=i)
        seq = rolled if seq is None else bundle(seq, rolled)
    seq.normalize()
    assert seq.size == D
    print("PASS hdlib-encoding-data")


# ---------------------------------------------------------------------------
# hdlib-reproducibility
# ---------------------------------------------------------------------------
def test_reproducibility():
    from hdlib.vector import Vector
    from hdlib.space import Space
    from hdlib.model import ClassificationModel, GraphModel, ClusteringModel
    from sklearn.datasets import make_blobs

    # Vector
    assert (Vector(seed=1).vector == Vector(seed=1).vector).all()

    # ClassificationModel with seed=0
    points = [[i, i * 0.5, i % 7] for i in range(60)]
    labels = ["a" if i % 2 == 0 else "b" for i in range(60)]
    m1 = ClassificationModel(size=2000, levels=5)
    m1.fit(points, labels=labels, seed=0)
    m2 = ClassificationModel(size=2000, levels=5)
    m2.fit(points, labels=labels, seed=0)
    v1 = m1.space.get(names=["level_0"])[0].vector
    v2 = m2.space.get(names=["level_0"])[0].vector
    assert (v1 == v2).all()

    # GraphModel - node vectors are NOT seeded by seed=...; we must
    # pre-build them deterministically.
    edges = {("a", "b", 1), ("b", "c", 1)}

    def make_graph():
        g = GraphModel(size=2000, directed=False, seed=0)
        for n in {"a", "b", "c"}:
            v = Vector(name=n, size=g.size, vtype=g.vtype, seed=hash(n) % (2**31))
            # GraphModel._add_edge attaches these two attributes to fresh vectors,
            # but since we are pre-inserting, we have to set them manually.
            setattr(v, "memory", None)
            setattr(v, "weights", {})
            g.space.insert(v)
        g.fit(edges)
        return g

    g1 = make_graph()
    g2 = make_graph()
    assert (g1.space.space["a"].vector == g2.space.space["a"].vector).all()

    # ClusteringModel - per-construction deterministic
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=0)
    c1 = ClusteringModel(k=3, n_features=4, size=1000, seed=0)
    c1.fit(X)
    c2 = ClusteringModel(k=3, n_features=4, size=1000, seed=0)
    c2.fit(X)
    assert (c1.labels_ == c2.labels_).all()
    print("PASS hdlib-reproducibility")


# ---------------------------------------------------------------------------
# hdlib-pitfalls: spot-check a few exception types
# ---------------------------------------------------------------------------
def test_pitfalls():
    from hdlib.vector import Vector
    from hdlib.space import Space
    from hdlib.arithmetic import bind, subtraction

    # Mismatched size
    a = Vector(size=10)
    b = Vector(size=20)
    try:
        bind(a, b)
    except Exception as e:
        assert "same size" in str(e)
    else:
        assert False, "expected Exception on size mismatch"

    # Mismatched vtype
    bin_a = Vector(size=10, vtype="binary")
    bip_a = Vector(size=10, vtype="bipolar")
    try:
        bind(bin_a, bip_a)
    except Exception as e:
        assert "types" in str(e).lower()
    else:
        assert False, "expected Exception on vtype mismatch"

    # Binary subtraction forbidden
    b1 = Vector(size=10, vtype="binary")
    b2 = Vector(size=10, vtype="binary")
    try:
        subtraction(b1, b2)
    except Exception as e:
        assert "binary" in str(e).lower()
    else:
        assert False, "expected Exception on binary subtraction"

    # Dump refuses to overwrite
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.pkl")
        v = Vector(name="v")
        v.dump(to_file=path)
        try:
            v.dump(to_file=path)
        except Exception as e:
            assert "already exists" in str(e)
        else:
            assert False, "expected Exception on overwrite"

    # Space size/vtype mismatch
    sp = Space(size=10, vtype="bipolar")
    try:
        sp.insert(Vector(size=15, vtype="bipolar"))
    except Exception as e:
        assert "size" in str(e).lower()
    else:
        assert False, "expected Exception on size mismatch in space"

    print("PASS hdlib-pitfalls")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running hdlib-skills smoke tests against hdlib", end=" ")
    import hdlib
    print(hdlib.__version__)
    print()

    tests = [
        test_overview,
        test_vectors,
        test_space,
        test_arithmetic,
        test_distance,
        test_classification,
        test_feature_selection,
        test_hyperparameter_tuning,
        test_clustering,
        test_regression,
        test_graph,
        test_quantum_arithmetic,
        test_quantum_classification,
        test_quantum_advanced,
        test_analogical_reasoning,
        test_encoding_data,
        test_reproducibility,
        test_pitfalls,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            failures.append((t.__name__, str(exc)))
    print()
    if failures:
        print(f"{len(failures)} test(s) failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} skill smoke tests passed.")
