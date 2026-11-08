"""LFA -- Label-Free Alignment (RQ4).

The hypothesis of the paper is that unlabeled data helps exactly when the
*label-free* geometry of the MiniRocket feature space is already aligned with
the (unknown) class structure.  LFA measures that geometry using **no labels
at all**.

Assumption stated in the paper: the number of classes ``C`` is known.  That is
realistic -- a practitioner planning to label ECG windows knows how many
diagnostic categories they intend to distinguish -- but it *is* an assumption
and it is the only label-derived quantity LFA uses.

Four raw components, all computed on ``F_{L u U}`` (the training features
produced by the MiniRocket transform fitted on ``L u U``):

``sil``               silhouette score of a C-means clustering
``chi``               log Calinski-Harabasz index of the same clustering
``stab``              cluster stability under bootstrap resampling (mean ARI)
``knn_purity_proxy``  mean fraction of a point's 7-NN sharing its cluster id

The composite ``LFA`` is the mean of the four components **z-scored across
datasets**.  The z-scoring is deliberately *not* done inside this module: a
z-score is only meaningful relative to a population of datasets, so the raw
components are stored per dataset and standardised later in the analysis stage
(:func:`composite_lfa`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

LFA_COMPONENTS = ("sil", "chi", "stab", "knn_purity_proxy")


@dataclass
class LFAComponents:
    sil: float
    chi: float
    stab: float
    knn_purity_proxy: float
    n_samples: int
    n_features_used: int

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _prepare(
    features: np.ndarray, pca_dims: int | None, seed: int
) -> np.ndarray:
    """Standardise then (optionally) PCA-reduce.

    Standardising first matters: MiniRocket PPV features have very unequal
    variances and an unscaled PCA would be dominated by a handful of kernels.
    """
    X = np.asarray(features, dtype=np.float64)
    X = StandardScaler().fit_transform(X)
    if pca_dims and X.shape[1] > pca_dims:
        n_comp = int(min(pca_dims, X.shape[0] - 1, X.shape[1]))
        if n_comp >= 2:
            X = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float64)


def _kmeans(X: np.ndarray, n_clusters: int, seed: int, n_init: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    return km.fit_predict(X)


def cluster_stability(
    X: np.ndarray,
    n_clusters: int,
    seed: int,
    n_bootstrap: int = 10,
    frac: float = 0.8,
    n_init: int = 10,
) -> float:
    """Mean pairwise ARI of cluster assignments over bootstrap subsamples.

    For every pair of subsamples we score the ARI **on their shared points**
    only, which is what makes the comparison well defined despite the two runs
    seeing different data.
    """
    n = X.shape[0]
    size = max(n_clusters + 1, int(round(frac * n)))
    rng = np.random.default_rng(seed)
    assignments: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(n_bootstrap):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        lab = _kmeans(X[idx], n_clusters, seed=seed + 1000 + b, n_init=n_init)
        assignments.append((idx, lab))

    aris: list[float] = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            idx_i, lab_i = assignments[i]
            idx_j, lab_j = assignments[j]
            shared, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if shared.size < n_clusters + 1:
                continue
            aris.append(float(adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])))
    return float(np.mean(aris)) if aris else float("nan")


def knn_cluster_purity(X: np.ndarray, labels: np.ndarray, k: int = 7) -> float:
    """Mean fraction of a point's ``k`` nearest neighbours sharing its cluster id."""
    n = X.shape[0]
    kk = int(min(k, n - 1))
    if kk < 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    _, ind = nn.kneighbors(X)
    neigh = ind[:, 1:]  # drop self
    same = labels[neigh] == labels[:, None]
    return float(np.mean(same))


def compute_lfa_components(
    features: np.ndarray,
    n_classes: int,
    seed: int = 0,
    pca_dims: int | None = 50,
    n_init: int = 10,
    n_bootstrap: int = 10,
    bootstrap_frac: float = 0.8,
    knn_k: int = 7,
) -> LFAComponents:
    """Compute the four raw LFA components.  **No labels are used.**"""
    X = _prepare(features, pca_dims, seed)
    n = X.shape[0]
    C = int(n_classes)
    if C < 2:
        raise ValueError("LFA needs at least 2 classes")
    if n <= C:
        raise ValueError(f"need more samples ({n}) than clusters ({C})")

    labels = _kmeans(X, C, seed=seed, n_init=n_init)
    n_found = len(np.unique(labels))

    if n_found < 2:
        sil = float("nan")
        chi = float("nan")
    else:
        sil = float(silhouette_score(X, labels, metric="euclidean"))
        raw_chi = float(calinski_harabasz_score(X, labels))
        # log-scaled per the specification; CH is unbounded above and heavily
        # right-skewed across datasets, which would otherwise dominate the mean.
        chi = float(np.log1p(max(raw_chi, 0.0)))

    stab = cluster_stability(
        X, C, seed=seed, n_bootstrap=n_bootstrap, frac=bootstrap_frac, n_init=n_init
    )
    purity = knn_cluster_purity(X, labels, k=knn_k)

    return LFAComponents(
        sil=sil,
        chi=chi,
        stab=stab,
        knn_purity_proxy=purity,
        n_samples=int(n),
        n_features_used=int(X.shape[1]),
    )


def zscore(values: np.ndarray) -> np.ndarray:
    """z-score across datasets; a zero-variance column becomes all-zeros."""
    v = np.asarray(values, dtype=float)
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def composite_lfa(component_table: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the z-scored components.

    ``component_table`` maps each name in :data:`LFA_COMPONENTS` to an array
    over datasets (same order).  z-scoring happens *here*, across datasets --
    never inside the per-dataset loop.
    """
    cols = []
    for name in LFA_COMPONENTS:
        if name not in component_table:
            raise KeyError(f"missing LFA component {name!r}")
        cols.append(zscore(np.asarray(component_table[name], dtype=float)))
    return np.nanmean(np.vstack(cols), axis=0)
