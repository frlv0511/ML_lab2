"""Часть 1: отбор признаков — корреляции, PCA, SVD, ICA, biplot."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import cross_val_score

from src.config import SEED

CLASS_COLORS = "viridis"


def correlation(df_x: pd.DataFrame):
    """1.1 Матрица взаимных корреляций (Пирсон) + тепловая карта."""
    corr = df_x.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns); ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax); ax.set_title("Матрица корреляций признаков")
    return corr, fig


def scatter_matrix(df_x: pd.DataFrame, y: np.ndarray):
    """1.2 Матрица разброса (цвет — эталонные метки y, только для визуализации)."""
    axes = pd.plotting.scatter_matrix(df_x, c=y, cmap=CLASS_COLORS, figsize=(9, 9),
                                      diagonal="hist", alpha=0.7, s=14)
    fig = axes[0, 0].get_figure()
    fig.suptitle("Матрица разброса признаков (цвет — метки y)", y=0.92)
    return fig


def run_pca(X):
    """1.3.1 PCA. Значимость: доля дисперсии и (кросс-валидационное) правдоподобие."""
    pca = PCA().fit(X)
    scores = pca.transform(X)
    ll = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(1, X.shape[1]):          # k = n_features вырожден (шум = 0)
            v = cross_val_score(PCA(n_components=k), X, cv=5).mean()
            if np.isfinite(v):                  # при k = rank шум ~ 0 -> правдоподобие не определено
                ll[k] = v
    return dict(model=pca, scores=scores, loadings=pca.components_.T,
                eigenvalues=pca.explained_variance_, ratio=pca.explained_variance_ratio_,
                loglik=ll)


def run_svd(X):
    """1.3.2 SVD центрированной матрицы: X = U S V^T. Значимость — s_i^2 / sum(s^2)."""
    U, s, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return dict(scores=U * s, loadings=Vt.T, s=s, ratio=s ** 2 / np.sum(s ** 2),
                eigenvalues=s ** 2 / (len(X) - 1))


def negentropy(z):
    """Приближение негэнтропии J(z) ~ (E[log cosh z] - E[log cosh v])^2, v ~ N(0,1)."""
    z = (z - z.mean(0)) / z.std(0)
    ref = np.mean(np.log(np.cosh(np.random.default_rng(SEED).standard_normal(200000))))
    return (np.log(np.cosh(z)).mean(0) - ref) ** 2


def run_ica(X, n_components):
    """1.3.3 ICA (FastICA). Значимость — негауссовость источников: негэнтропия и эксцесс."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n_components, whiten="unit-variance", random_state=SEED,
                      max_iter=2000, tol=1e-5)
        S = ica.fit_transform(X)
    J, kurt = negentropy(S), kurtosis(S, axis=0)
    order = np.argsort(-J)                          # от самой негауссовой к самой гауссовой
    S, mixing = S[:, order], ica.mixing_[:, order]
    return dict(model=ica, scores=S, loadings=mixing / np.abs(mixing).max(axis=0),
                negentropy=J[order], kurtosis=kurt[order], order=order,
                ratio=(mixing ** 2).sum(0) / (mixing ** 2).sum())


def mp_bounds(n, p):
    """Границы Марченко–Пастура для собственных значений корреляционной матрицы
    p некоррелированных стандартизованных признаков, n наблюдений."""
    q = np.sqrt(p / n)
    return (1 - q) ** 2, (1 + q) ** 2


def classify(eigenvalues, n):
    """Деление компонент на главные / второстепенные / шумовые по Марченко–Пастуру:
    lam > lam+ — главная (выделяется над шумом); lam- <= lam <= lam+ — второстепенная
    (неотличима от случайной выборочной вариации); lam < lam- — шумовая (вырожденная)."""
    lo, hi = mp_bounds(n, len(eigenvalues))
    return ["главная" if v > hi else "второстепенная" if v >= lo else "шумовая"
            for v in eigenvalues], lo, hi


def plot_significance(pca, svd, ica, n):
    """1.4 Графики значимости компонент для PCA, SVD, ICA."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    k = np.arange(1, len(pca["ratio"]) + 1)
    lo, hi = mp_bounds(n, len(k))
    ax = axes[0, 0]
    ax.bar(k, pca["eigenvalues"], color="tab:blue", alpha=0.8, label="собств. значения")
    ax.axhspan(lo, hi, color="gray", alpha=0.25, label="область шума (Марченко–Пастур)")
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set_title("PCA: собственные значения"); ax.set_xlabel("Компонента"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.bar(k, pca["ratio"] * 100, color="tab:blue", alpha=0.8)
    ax.plot(k, np.cumsum(pca["ratio"]) * 100, "ro-", label="накопленная")
    ax.set_title("PCA: объяснённая дисперсия, %"); ax.set_xlabel("Компонента"); ax.legend()
    ax = axes[0, 1]
    ax.semilogy(k, svd["s"] + 1e-16, "s-", color="tab:green")
    ax.set_title("SVD: сингулярные числа (лог. шкала)"); ax.set_xlabel("Компонента"); ax.grid(alpha=.3)
    ax = axes[1, 1]
    ax.bar(k, svd["ratio"] * 100, color="tab:green", alpha=0.8)
    ax.plot(k, np.cumsum(svd["ratio"]) * 100, "ro-")
    ax.set_title("SVD: доля s²/Σs², %"); ax.set_xlabel("Компонента")
    ax = axes[0, 2]
    kk = np.arange(1, len(ica["negentropy"]) + 1)
    ax.bar(kk, ica["negentropy"], color="tab:orange", alpha=0.85)
    ax.set_title("ICA: негэнтропия независимых компонент"); ax.set_xlabel("Компонента (по убыванию)")
    ax = axes[1, 2]
    ll_k = list(pca["loglik"]); ax.plot(ll_k, [pca["loglik"][i] for i in ll_k], "mo-")
    ax.set_title("PCA: правдоподобие (CV, 5 блоков)"); ax.set_xlabel("Число компонент k")
    ax.set_ylabel("Среднее log-правдоподобие"); ax.grid(alpha=.3)
    from matplotlib.ticker import MaxNLocator
    for a in axes.ravel():
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def biplot(ax, scores, loadings, labels, y, title, xlab, ylab):
    """1.5 Проекция на 2 главные компоненты + стрелки признаков (по образцу из методички)."""
    sx = 1.0 / (scores[:, 0].max() - scores[:, 0].min())
    sy = 1.0 / (scores[:, 1].max() - scores[:, 1].min())
    sc = ax.scatter(scores[:, 0] * sx, scores[:, 1] * sy, c=y, cmap=CLASS_COLORS,
                    alpha=0.7, edgecolors="k", linewidths=0.4, s=25)
    for i, name in enumerate(labels):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color="r", alpha=0.7,
                 head_width=0.03, head_length=0.03, linewidth=1.5)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, name, color="darkred",
                ha="center", va="center", fontweight="bold")
    ax.axhline(0, color="gray", ls="--", lw=0.5); ax.axvline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.grid(True, ls=":", alpha=0.6)
    ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    return sc


def plot_biplots(pca, svd, ica, labels, y):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sc = biplot(axes[0], pca["scores"], pca["loadings"], labels, y, "PCA biplot",
                f"PC1 ({pca['ratio'][0]*100:.1f}% дисперсии)", f"PC2 ({pca['ratio'][1]*100:.1f}%)")
    biplot(axes[1], svd["scores"], svd["loadings"], labels, y, "SVD biplot",
           f"SV1 ({svd['ratio'][0]*100:.1f}%)", f"SV2 ({svd['ratio'][1]*100:.1f}%)")
    biplot(axes[2], ica["scores"], ica["loadings"], labels, y, "ICA biplot",
           f"IC1 (негэнтропия {ica['negentropy'][0]:.4f})", f"IC2 ({ica['negentropy'][1]:.4f})")
    handles, _ = sc.legend_elements()
    axes[0].legend(handles, [int(v) for v in np.unique(y)], title="Класс y", loc="lower left")
    fig.tight_layout()
    return fig


def rank_and_dependencies(X, names):
    """Ранг матрицы (по SVD) и R^2 регрессии каждого признака на остальные:
    R^2 = 1 означает, что признак — точная линейная комбинация остальных."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    rank = int((s > 1e-8 * s[0]).sum())
    r2 = {}
    for j, nm in enumerate(names):
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(np.c_[others, np.ones(len(X))], X[:, j], rcond=None)
        res = X[:, j] - np.c_[others, np.ones(len(X))] @ beta
        r2[nm] = 1 - res.var() / X[:, j].var()
    return rank, r2


def classify_ica(kurt):
    """Негауссовость источника: |эксцесс| > 1 — главная, < 0.5 — шумовая (гауссова)."""
    return ["главная" if abs(k) > 1 else "шумовая" if abs(k) < 0.5 else "второстепенная"
            for k in kurt]
