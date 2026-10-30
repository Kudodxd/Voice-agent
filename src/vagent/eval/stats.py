from __future__ import annotations

import numpy as np


def paired_cluster_bootstrap(err_a, n_a, err_b, n_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Clusters (= sessions) are resampled jointly. Returns delta = rate(B) - rate(A)."""
    err_a, n_a, err_b, n_b = map(lambda x: np.asarray(x, float), (err_a, n_a, err_b, n_b))
    rng = np.random.default_rng(seed)
    k = len(err_a)
    idx = rng.integers(0, k, size=(n_boot, k))
    da = err_a[idx].sum(1) / np.maximum(n_a[idx].sum(1), 1)
    db = err_b[idx].sum(1) / np.maximum(n_b[idx].sum(1), 1)
    d = db - da
    delta = err_b.sum() / max(n_b.sum(), 1) - err_a.sum() / max(n_a.sum(), 1)
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    return {"delta": float(delta), "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p": float(min(1.0, p)), "n_clusters": k}
