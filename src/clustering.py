"""Часть 2: 12 алгоритмов кластеризации, подбор параметров, аномалии, сравнение с эталоном."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster as C
from sklearn import metrics as M
from sklearn.mixture import GaussianMixture

from src.config import SEED

NAMES = ["K-Means", "Affinity propagation", "Mean-shift", "Spectral clustering",
         "Ward hierarchical", "Agglomerative", "DBSCAN", "HDBSCAN", "OPTICS",
         "Gaussian mixtures", "BIRCH", "Bisecting K-Means"]


def algorithms(X, tuned: bool) -> dict:
    """Алгоритмы с исходными (tuned=False) или подобранными (tuned=True) параметрами."""
    if not tuned:
        return {
            "K-Means": C.KMeans(3, n_init=10, random_state=SEED),
            "Affinity propagation": C.AffinityPropagation(random_state=SEED),
            "Mean-shift": C.MeanShift(),
            "Spectral clustering": C.SpectralClustering(3, random_state=SEED),
            "Ward hierarchical": C.AgglomerativeClustering(3, linkage="ward"),
            "Agglomerative": C.AgglomerativeClustering(3, linkage="single"),
            "DBSCAN": C.DBSCAN(),                      # eps=0.5, min_samples=5
            "HDBSCAN": C.HDBSCAN(),
            "OPTICS": C.OPTICS(),
            "Gaussian mixtures": GaussianMixture(3, random_state=SEED),
            "BIRCH": C.Birch(n_clusters=3, threshold=0.1),
            "Bisecting K-Means": C.BisectingKMeans(3, random_state=SEED),
        }
    return {
        "K-Means": C.KMeans(3, n_init=10, random_state=SEED),
        "Affinity propagation": C.AffinityPropagation(damping=0.9, preference=-20, random_state=SEED),
        "Mean-shift": C.MeanShift(bandwidth=float(C.estimate_bandwidth(X, quantile=0.3))),
        "Spectral clustering": C.SpectralClustering(3, affinity="nearest_neighbors", n_neighbors=10,
                                                    random_state=SEED),
        "Ward hierarchical": C.AgglomerativeClustering(3, linkage="ward"),
        "Agglomerative": C.AgglomerativeClustering(3, linkage="average"),
        "DBSCAN": C.DBSCAN(eps=0.4, min_samples=5),
        "HDBSCAN": C.HDBSCAN(min_cluster_size=10),
        "OPTICS": C.OPTICS(min_samples=5, cluster_method="dbscan", eps=0.4),
        "Gaussian mixtures": GaussianMixture(3, covariance_type="full", random_state=SEED),
        "BIRCH": C.Birch(n_clusters=3, threshold=0.5),
        "Bisecting K-Means": C.BisectingKMeans(3, random_state=SEED),
    }


def fit_all(X, tuned: bool) -> dict:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, est in algorithms(X, tuned).items():
            out[name] = np.asarray(est.fit_predict(X))
    return out


def n_clusters(lab):
    return len(set(lab)) - (1 if -1 in lab else 0)


def plot_grid(X, labels: dict, title: str):
    """2.2 / 2.3 Результаты всех алгоритмов на одном графике (plt.subplots)."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    cmap = plt.get_cmap("tab10")
    for ax, name in zip(axes.ravel(), NAMES):
        lab = labels[name]
        for l in sorted(set(lab)):
            m = lab == l
            if l == -1:
                ax.scatter(X[m, 0], X[m, 1], c="k", marker="x", s=30, label="выбросы")
            else:
                ax.scatter(X[m, 0], X[m, 1], color=cmap(l % 10), s=14, edgecolor="k", linewidths=0.2)
        noise = int((lab == -1).sum())
        ax.set_title(f"{name}\nкластеров: {n_clusters(lab)}" + (f", выбросов: {noise}" if noise else ""),
                     fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=14); fig.tight_layout()
    return fig


def internal_quality(X, lab):
    """Silhouette по точкам, отнесённым к кластерам (выбросы -1 не учитываются)."""
    m = lab != -1
    if n_clusters(lab) < 2 or m.sum() < 3:
        return np.nan
    return M.silhouette_score(X[m], lab[m])


def scan_kmeans_gmm(X, ks=range(2, 9)):
    """Выбор числа кластеров: инерция и silhouette (K-Means), BIC (GMM)."""
    rows = []
    for k in ks:
        km = C.KMeans(k, n_init=10, random_state=SEED).fit(X)
        gm = GaussianMixture(k, random_state=SEED).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=M.silhouette_score(X, km.labels_),
                         bic=gm.bic(X)))
    df = pd.DataFrame(rows).set_index("k")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df.index, df.inertia, "o-"); axes[0].set_title("Метод локтя (K-Means): инерция")
    axes[1].plot(df.index, df.silhouette, "o-", color="tab:green"); axes[1].set_title("Silhouette (K-Means)")
    axes[2].plot(df.index, df.bic, "o-", color="tab:red"); axes[2].set_title("BIC (Gaussian mixtures)")
    for a in axes:
        a.set_xlabel("Число кластеров k"); a.grid(alpha=0.3)
    fig.tight_layout()
    return df, fig


def scan_density(X):
    """Таблицы подбора параметров для DBSCAN, HDBSCAN, OPTICS по внутренним критериям
    (число кластеров, доля выбросов, silhouette) — метки y не используются."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eps in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            for ms in (3, 5, 8):
                lab = C.DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                rows.append(dict(метод="DBSCAN", параметры=f"eps={eps}, min_samples={ms}",
                                 кластеров=n_clusters(lab), выбросов=int((lab == -1).sum()),
                                 silhouette=internal_quality(X, lab)))
        for mcs in (5, 8, 10, 15, 20):
            lab = C.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            rows.append(dict(метод="HDBSCAN", параметры=f"min_cluster_size={mcs}",
                             кластеров=n_clusters(lab), выбросов=int((lab == -1).sum()),
                             silhouette=internal_quality(X, lab)))
        for ms in (5, 10, 15):
            for xi in (0.02, 0.05, 0.1):
                lab = C.OPTICS(min_samples=ms, xi=xi, min_cluster_size=0.1).fit_predict(X)
                rows.append(dict(метод="OPTICS", параметры=f"min_samples={ms}, xi={xi}",
                                 кластеров=n_clusters(lab), выбросов=int((lab == -1).sum()),
                                 silhouette=internal_quality(X, lab)))
        for ms in (3, 5, 8):
            for eps in (0.25, 0.3, 0.4, 0.5):
                lab = C.OPTICS(min_samples=ms, cluster_method="dbscan", eps=eps).fit_predict(X)
                rows.append(dict(метод="OPTICS (dbscan-извлечение)", параметры=f"min_samples={ms}, eps={eps}",
                                 кластеров=n_clusters(lab), выбросов=int((lab == -1).sum()),
                                 silhouette=internal_quality(X, lab)))
    return pd.DataFrame(rows)


def plot_density_diagnostics(X, k=5):
    """График k-расстояний (выбор eps для DBSCAN) и график достижимости OPTICS."""
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    kd = np.sort(d[:, -1])[::-1]
    opt = C.OPTICS(min_samples=k).fit(X)
    reach = opt.reachability_[opt.ordering_]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(kd); axes[0].axhline(0.4, color="r", ls="--", label="eps = 0.4")
    axes[0].set_title(f"DBSCAN: расстояние до {k}-го соседа (по убыванию)")
    axes[0].set_xlabel("Объекты"); axes[0].set_ylabel("расстояние"); axes[0].legend(); axes[0].grid(alpha=.3)
    r = np.where(np.isinf(reach), np.nan, reach)
    axes[1].plot(r, lw=1); axes[1].axhline(0.4, color="r", ls="--", label="порог eps = 0.4")
    axes[1].set_title("OPTICS: график достижимости"); axes[1].set_xlabel("Порядок обхода")
    axes[1].set_ylabel("расстояние достижимости"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def compare_with_truth(X, y, labels: dict) -> pd.DataFrame:
    """2.5 Сравнение с эталоном (метки y используются только здесь)."""
    rows = []
    for name in NAMES:
        lab = labels[name]
        rows.append({
            "Алгоритм": name, "Кластеров": n_clusters(lab), "Выбросов": int((lab == -1).sum()),
            "ARI": M.adjusted_rand_score(y, lab), "NMI": M.normalized_mutual_info_score(y, lab),
            "Однородность": M.homogeneity_score(y, lab), "Полнота": M.completeness_score(y, lab),
            "V-мера": M.v_measure_score(y, lab), "Silhouette": internal_quality(X, lab)})
    return pd.DataFrame(rows).set_index("Алгоритм")


def plot_anomalies(X, labels: dict, names=("DBSCAN", "HDBSCAN", "OPTICS")):
    """2.4 Кластеры + кластер аномальных данных (метка -1) для плотностных методов."""
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    cmap = plt.get_cmap("tab10")
    for ax, name in zip(np.atleast_1d(axes), names):
        lab = labels[name]
        for l in sorted(set(lab)):
            m = lab == l
            if l == -1:
                ax.scatter(X[m, 0], X[m, 1], c="red", marker="X", s=90, edgecolor="k",
                           label=f"аномалии ({m.sum()})", zorder=3)
            else:
                ax.scatter(X[m, 0], X[m, 1], color=cmap(l % 10), s=18, edgecolor="k",
                           linewidths=0.3, label=f"кластер {l + 1} ({m.sum()})")
        ax.set_title(f"{name}: выделение аномалий"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_truth_vs_best(X, y, lab, best):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for l in np.unique(y):
        m = y == l
        axes[0].scatter(X[m, 0], X[m, 1], color=cmap(int(l)), s=18, edgecolor="k", linewidths=0.3,
                        label=f"класс {int(l)}")
    axes[0].set_title("Эталонные метки y"); axes[0].legend()
    # цвета кластеров согласуем с классами y (венгерский алгоритм) — только для наглядности
    from scipy.optimize import linear_sum_assignment
    cls, cl_ids = np.unique(y), [l for l in sorted(set(lab)) if l != -1]
    cont = np.array([[np.sum((lab == l) & (y == c)) for c in cls] for l in cl_ids])
    r, c = linear_sum_assignment(-cont)
    color_of = {cl_ids[i]: int(cls[j]) for i, j in zip(r, c)}
    for l in sorted(set(lab)):
        m = lab == l
        axes[1].scatter(X[m, 0], X[m, 1], color="k" if l == -1 else cmap(color_of.get(l, int(l)) % 10),
                        marker="x" if l == -1 else "o", s=18, edgecolor="k", linewidths=0.3)
    axes[1].set_title(f"Результат: {best}")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig
