"""A check of search_lab.qpbo_binary (roof duality by one max flow on the doubled graph) on small random binary
energies, against brute force:
  * submodular energies: every node labelled, and the labelling is a minimiser;
  * any energy: persistency (fixing the labelled nodes and minimising the rest by brute force reaches the global
    minimum) and autarky (the labelled nodes over the all-zero labelling never raise its energy).

    uv run python scripts/incidents/search_qpbo_check.py"""
import itertools

import numpy as np

import search_lab as L


def energy(c0, c1, pa, pb, e, s):
    pair = e[np.arange(len(pa)), 2 * s[pa] + s[pb]]
    return float(np.where(s, c1, c0).sum() + pair.sum())


def main(trials=400, n=9, seed=0):
    rng = np.random.default_rng(seed)
    stats = {"labelled": 0, "nodes": 0, "submodular_ok": 0, "persistent": 0, "autarky": 0}
    for t in range(trials):
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35])
        if not len(pairs):
            continue
        pa, pb = pairs[:, 0], pairs[:, 1]
        c0, c1 = rng.normal(0, 1, n), rng.normal(0, 1, n)
        e = rng.exponential(1.0, (len(pa), 4))          # e00 e01 e10 e11
        submodular = t % 2 == 0
        if submodular:                                   # e00 + e11 <= e01 + e10
            e[:, 3] = np.minimum(e[:, 3], e[:, 1] + e[:, 2] - e[:, 0])
        allS = np.array(list(itertools.product([0, 1], repeat=n)), bool)
        Es = np.array([energy(c0, c1, pa, pb, e, s.astype(int)) for s in allS])
        best = Es.min()
        s, lab = L.qpbo_binary(c0, c1, pa, pb, e[:, 0], e[:, 1], e[:, 2], e[:, 3])
        stats["labelled"] += int(lab.sum())
        stats["nodes"] += n
        if submodular:
            assert lab.all(), f"trial {t}: a submodular energy left nodes unlabelled"
            assert abs(energy(c0, c1, pa, pb, e, s.astype(int)) - best) < 1e-9, f"trial {t}: not a minimiser"
            stats["submodular_ok"] += 1
        ok = np.all(allS[:, lab] == s[lab], axis=1)
        assert abs(Es[ok].min() - best) < 1e-9, f"trial {t}: the labelled nodes are not persistent"
        stats["persistent"] += 1
        z = np.zeros(n, bool)
        z[lab] = s[lab]
        assert energy(c0, c1, pa, pb, e, z.astype(int)) <= Es[0] + 1e-9, f"trial {t}: autarky fails"
        stats["autarky"] += 1
    print(stats, f"share labelled {stats['labelled'] / stats['nodes']:.1%}")


if __name__ == "__main__":
    main()
