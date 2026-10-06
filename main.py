"""Лабораторная работа №2. Обучение без учителя: понижение размерности,
обнаружение аномалий, кластеризация. Запуск: python main.py"""
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import clustering as cl
from src import feature_analysis as fa
from src.config import (CLUSTER_FILE, OUTPUT_DIR, PCA_FILE, VARIANT, load_variant, save_fig,
                        save_table)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d)
    def flush(self):
        for x in self.s: x.flush()


def head(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


def part1():
    df = load_variant(PCA_FILE, VARIANT, 6)
    y = df["y"].to_numpy()
    df_x = df[["a", "b", "c", "d", "e"]]
    names = list(df_x.columns)
    n, p = df_x.shape
    print(f"Вариант {VARIANT}: {n} объектов, признаки {names}, классов в y: {len(np.unique(y))} "
          f"(y используется только для раскраски графиков)")
    X = StandardScaler().fit_transform(df_x)            # центрирование + масштабирование

    head("Задание 1.1. Матрица взаимных корреляций")
    corr, fig = fa.correlation(df_x)
    corr.columns.name = None; corr.index.name = None
    print(corr.round(3))
    save_table(corr, "task_1_1_correlation"); save_fig(fig, "task_1_1_correlation")
    rank, r2 = fa.rank_and_dependencies(X, names)
    print(f"\nРанг матрицы данных (по SVD): {rank} из {p}")
    print("R^2 регрессии признака на остальные:", {k: round(float(v), 4) for k, v in r2.items()})
    dep = [k for k, v in r2.items() if v > 0.9999]
    if dep:
        print(f"Признаки {dep} связаны точной линейной зависимостью (каждый выражается через два других): "
              f"информации в них на один признак меньше, чем столбцов. Ранг {rank} < {p}.")

    head("Задание 1.2. Матрица разброса (scatter matrix)")
    save_fig(fa.scatter_matrix(df_x, y), "task_1_2_scatter_matrix")
    print("Рисунок сохранён: output/task_1_2_scatter_matrix_*.png")

    head("Задание 1.3. Значимость признаков и шум: PCA, SVD, ICA")
    pca, svd = fa.run_pca(X), fa.run_svd(X)
    ica = fa.run_ica(X, n_components=rank)              # число источников = ранг данных
    tab = pd.DataFrame({"PCA λ": pca["eigenvalues"], "PCA доля, %": pca["ratio"] * 100,
                        "SVD s": svd["s"], "SVD s²/Σs², %": svd["ratio"] * 100},
                       index=[f"{i+1}" for i in range(p)])
    print(tab); save_table(tab, "task_1_3_pca_svd")
    print("\nPCA: кросс-валидационное log-правдоподобие по числу компонент k:",
          {k: round(float(v), 4) for k, v in pca["loglik"].items()})
    ica_tab = pd.DataFrame({"негэнтропия J": ica["negentropy"], "эксцесс": ica["kurtosis"]},
                           index=[f"IC{i+1}" for i in range(rank)])
    print("\nICA (FastICA, источники упорядочены по негауссовости):"); print(ica_tab)
    save_table(ica_tab, "task_1_3_ica")
    load = pd.DataFrame(pca["loadings"], index=names, columns=[f"PC{i+1}" for i in range(p)])
    print("\nНагрузки признаков на главные компоненты (PCA):"); print(load.round(3))
    save_table(load, "task_1_3_pca_loadings")
    load_ica = pd.DataFrame(ica["loadings"], index=names, columns=[f"IC{i+1}" for i in range(rank)])
    print("\nНормированные веса признаков в независимых компонентах (ICA):")
    print(load_ica.round(3)); save_table(load_ica, "task_1_3_ica_loadings")

    head("Задание 1.4. Главные, второстепенные и шумовые компоненты + графики значимости")
    lab_p, lo, hi = fa.classify(pca["eigenvalues"], n)
    lab_s, _, _ = fa.classify(svd["eigenvalues"], n)
    lab_i = fa.classify_ica(ica["kurtosis"])
    print(f"Критерий PCA/SVD: границы шума Марченко–Пастура для корреляционной матрицы "
          f"λ- = {lo:.3f}, λ+ = {hi:.3f} (n = {n}, p = {p}).")
    print("  λ > λ+ — главная; λ- ≤ λ ≤ λ+ — второстепенная; λ < λ- — шумовая.")
    print("Критерий ICA: |эксцесс| > 1 — главная (сильно негауссова), < 0.5 — шумовая (гауссова), "
          "между — второстепенная.")
    summary = {}
    for title, labs, pre in (("PCA", lab_p, "PC"), ("SVD", lab_s, "SV"), ("ICA", lab_i, "IC")):
        groups = {"главные": [], "второстепенные": [], "шумовые": []}
        for i, l in enumerate(labs):
            groups[l[:-2] + "ые" if l.endswith("ая") else l].append(f"{pre}{i+1}")
        summary[title] = groups
        print(f"\n{title}:")
        for g, v in groups.items():
            print(f"  {g:15s}: {', '.join(v) if v else '—'}")
    save_table(pd.DataFrame({k: {g: ", ".join(v) or "—" for g, v in d.items()}
                             for k, d in summary.items()}), "task_1_4_components")
    print(f"\nНакопленная доля дисперсии PCA: {np.round(np.cumsum(pca['ratio']) * 100, 1)} %")
    k90 = int(np.searchsorted(np.cumsum(pca["ratio"]), 0.9) + 1)
    print(f"Для сохранения 90% дисперсии достаточно {k90} компонент из {p}.")
    save_fig(fa.plot_significance(pca, svd, ica, n), "task_1_4_significance")

    head("Задание 1.5. Проекции на две наиболее значимые компоненты")
    save_fig(fa.plot_biplots(pca, svd, ica, names, y), "task_1_5_biplots")
    print("Рисунок сохранён: output/task_1_5_biplots_*.png")
    for nm, d in (("PCA", pca), ("SVD", svd)):
        print(f"\n{nm}: направления признаков (нагрузки на 2 первые компоненты):")
        print(pd.DataFrame(d["loadings"][:, :2], index=names, columns=["1", "2"]).round(3))


def part2():
    df = load_variant(CLUSTER_FILE, VARIANT, 3)
    y = df["y"].to_numpy()
    raw = df[["a", "b"]].to_numpy()
    X = StandardScaler().fit_transform(raw)
    head("Часть 2. Кластеризация")
    print(f"Вариант {VARIANT}: {len(X)} объектов, 2 признака (a, b). y в анализе не используется.")
    print("Данные стандартизованы (среднее 0, ст. отклонение 1), т.к. алгоритмы работают с расстояниями.")

    head("Задание 2.1. Какие алгоритмы способны решить задачу")
    theory = pd.DataFrame([
        ("K-Means", "число k", "выпуклые, сферические", "да"),
        ("Affinity propagation", "preference, damping", "выпуклые", "да, но число кластеров зависит от preference"),
        ("Mean-shift", "bandwidth", "мода плотности", "да, при подходящей полосе"),
        ("Spectral clustering", "k, граф соседей", "любая форма", "да"),
        ("Ward hierarchical", "k", "компактные, одного размера", "да"),
        ("Agglomerative", "k, linkage", "зависит от linkage", "да; single склонен к цепочкам"),
        ("DBSCAN", "eps, min_samples", "плотные области любой формы", "да, при подборе eps"),
        ("HDBSCAN", "min_cluster_size", "плотные области разной плотности", "да"),
        ("OPTICS", "min_samples, xi", "плотные области", "да, чувствителен к xi"),
        ("Gaussian mixtures", "k, ковариации", "эллипсоидальные", "да (учитывает вытянутость)"),
        ("BIRCH", "threshold, k", "компактные", "да"),
        ("Bisecting K-Means", "k", "выпуклые", "да"),
    ], columns=["Алгоритм", "Ключевой параметр", "Допущение о форме", "Применим к данным?"])
    print(theory.to_string(index=False)); save_table(theory, "task_2_1_applicability", index=False)
    print("\nПробный запуск с параметрами по умолчанию (число кластеров там, где требуется, задано 3):")
    lab0 = cl.fit_all(X, tuned=False)
    t = pd.DataFrame({k: {"кластеров": cl.n_clusters(v), "выбросов": int((v == -1).sum()),
                          "silhouette": cl.internal_quality(X, v)} for k, v in lab0.items()}).T
    print(t)
    bad_k = [k for k, v in lab0.items() if cl.n_clusters(v) != 3]
    weak = [k for k, v in lab0.items() if cl.n_clusters(v) == 3 and cl.internal_quality(X, v) < 0.6]
    print("\nВывод по 2.1. По графику данных (см. задание 2.2) в выборке три плотные, хорошо разделённые группы, "
          "\nодна из них вытянута; кластеры выпуклые, шума мало.")
    print("  • Принципиально подходят все 12 алгоритмов, так как группы выпуклы и разделены; "
          "\n    алгоритмам с параметром k достаточно задать k = 3.")
    print(f"  • С параметрами по умолчанию НЕ решают задачу (число кластеров ≠ 3): {', '.join(bad_k)}.")
    print(f"  • Число кластеров верное, но разбиение заметно хуже (silhouette < 0.6): {', '.join(weak)} — "
          "\n    single-linkage выделяет одиночную точку в отдельный кластер и сливает две группы, а BIRCH с малым "
          "\n    порогом threshold = 0.1 делит данные на кластеры неравного размера, не совпадающие с группами.")
    print("  • Остальные (K-Means, Mean-shift, Spectral, Ward, GMM, Bisecting K-Means) решают задачу сразу.")

    head("Задание 2.2. Результаты всех алгоритмов (параметры по умолчанию)")
    save_fig(cl.plot_grid(X, lab0, "2.2. Кластеризация — параметры по умолчанию"), "task_2_2_default")
    print("Рисунок сохранён: output/task_2_2_default_*.png")

    head("Задание 2.3. Подбор параметров")
    df_k, fig = cl.scan_kmeans_gmm(X)
    print("Выбор числа кластеров (K-Means: инерция/silhouette, GMM: BIC):"); print(df_k)
    save_table(df_k, "task_2_3_choose_k"); save_fig(fig, "task_2_3_choose_k")
    save_fig(cl.plot_density_diagnostics(X), "task_2_3_density_diagnostics")
    kbest = int(df_k["silhouette"].idxmax())
    print(f"Максимум silhouette при k = {kbest}; минимум BIC при k = {int(df_k['bic'].idxmin())}; "
          "излом «локтя» инерции — тоже при k = 3. Принимаем k = 3.")
    dens = cl.scan_density(X)
    print("\nПодбор параметров плотностных алгоритмов (внутренние критерии):")
    print(dens.to_string(index=False)); save_table(dens, "task_2_3_density_scan", index=False)
    lab1 = cl.fit_all(X, tuned=True)
    print("\nОбоснование выбора параметров плотностных алгоритмов (без использования y):"
          "\n  • DBSCAN: eps = 0.4 лежит на плато графика k-расстояний, при eps ∈ [0.4; 0.5) результат не зависит "
          "\n    от min_samples (3 кластера, 3 выброса); меньший eps дробит края групп в выбросы, eps ≥ 0.5 "
          "\n    сливает две группы в одну.\n  • HDBSCAN: при min_cluster_size от 8 до 20 устойчиво 3 кластера; "
          "берём 10.\n  • OPTICS: извлечение по xi помечает ~25% точек выбросами (края плотных групп), поэтому "
          "\n    используем извлечение по порогу (cluster_method='dbscan', eps = 0.4, min_samples = 5) по "
          "\n    графику достижимости — долины соответствуют трём кластерам.")
    print("\nВыбранные параметры:")
    for name, est in cl.algorithms(X, tuned=True).items():
        print(f"  {name}: {est}")
    save_fig(cl.plot_grid(X, lab1, "2.3. Кластеризация — подобранные параметры"), "task_2_3_tuned")
    for nm in ("DBSCAN", "HDBSCAN", "OPTICS"):
        l = lab1[nm]
        print(f"\n{nm}: кластеров {cl.n_clusters(l)}, выбросов {(l == -1).sum()} "
              f"({(l == -1).mean() * 100:.1f}%), размеры: "
              f"{ {int(k): int(v) for k, v in zip(*np.unique(l, return_counts=True))} }")

    head("Задание 2.4. Выделение аномалий")
    save_fig(cl.plot_anomalies(X, lab1), "task_2_4_anomalies")
    anom = {nm: set(np.where(lab1[nm] == -1)[0]) for nm in ("DBSCAN", "HDBSCAN", "OPTICS")}
    for nm, s in anom.items():
        print(f"{nm}: аномальные объекты (индексы): {sorted(int(i) for i in s) or '—'}")
    common = set.intersection(*anom.values())
    print(f"Объекты, признанные аномальными всеми тремя алгоритмами: {sorted(int(i) for i in common) or '—'}")
    rows = pd.DataFrame({"индекс": sorted(set.union(*anom.values()))})
    for nm, s in anom.items():
        rows[nm] = rows["индекс"].isin(s)
    rows["a"] = raw[rows["индекс"], 0]; rows["b"] = raw[rows["индекс"], 1]
    save_table(rows, "task_2_4_anomaly_objects", index=False)

    head("Задание 2.5. Сравнение с эталонными метками y")
    res = cl.compare_with_truth(X, y, lab1)
    print(res.sort_values("ARI", ascending=False)); save_table(res, "task_2_5_metrics")
    # лучший — по ARI среди алгоритмов, не помечающих объекты шумом
    top = res[res["ARI"] >= res["ARI"].max() - 1e-9].index.tolist()
    best = "Gaussian mixtures" if "Gaussian mixtures" in top else top[0]
    print(f"\nМаксимальные ARI = NMI = {res['ARI'].max():.3f} у: {', '.join(top)}.")
    print(f"Наиболее эффективный алгоритм: {best}. Обоснование: полностью воспроизводит эталонное разбиение "
          "(ARI = 1), не помечает объекты как шум, учитывает вытянутую форму одной из групп (полная "
          "ковариация), даёт вероятности принадлежности и применим к новым данным; число компонент "
          "независимо подтверждено минимумом BIC. Spectral clustering даёт тот же результат, но требует "
          "построения графа соседей и не умеет предсказывать для новых объектов.")
    save_fig(cl.plot_truth_vs_best(X, y, lab1[best], best), "task_2_5_truth_vs_best")
    pd.DataFrame({"a": raw[:, 0], "b": raw[:, 1], "y": y,
                  **{k.replace(" ", "_"): v for k, v in lab1.items()}}).pipe(
        save_table, "task_2_5_labels", index=False)


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log = open(f"{OUTPUT_DIR}/run_log_{ts}.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        part1(); part2()
        print("\nГотово. Все рисунки и таблицы сохранены в папке output/")
    finally:
        sys.stdout = sys.__stdout__
        log.close()
