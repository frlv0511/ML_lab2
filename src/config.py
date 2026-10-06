"""Конфигурация: вариант, seed, пути, сохранение результатов."""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANT = 15
SEED = 42
PCA_FILE = "data/Lab2_Dataset_PCA.xlsx"          # 24 варианта по 6 столбцов (a-e, y)
CLUSTER_FILE = "data/Lab2_Dataset_Clustering.xlsx"  # 24 варианта по 3 столбца (a, b, y)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_variant(path: str, variant: int, width: int) -> pd.DataFrame:
    """Вырезает из общего листа блок выбранного варианта (width столбцов)."""
    raw = pd.read_excel(path, header=None)
    block = raw.iloc[1:, (variant - 1) * width: variant * width].dropna(how="all")
    block.columns = block.iloc[0]
    return block.iloc[1:].astype(float).reset_index(drop=True)


def _name(base: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")


def save_fig(fig, base: str) -> str:
    path = _name(base, "png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, base: str, index: bool = True) -> str:
    path = _name(base, "csv")
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path
