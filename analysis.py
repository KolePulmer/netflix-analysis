from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def explode_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """India, Japan,.. неудобно машине - надо просплитить"""
    df = df.copy()
    df[col] = df[col].str.split(",")
    df = df.explode(col)
    df[col] = df[col].str.strip()
    df = df.dropna(subset=[col])  # убирает строки, где страна отсутствует
    df = df.loc[df[col] != ""]  # False в маске убирает пустые строки
    return df.reset_index(drop=True)  # старая индексация больше не актуальна


def iqr_outliers(series: pd.Series) -> tuple[pd.Series, float, float]:
    """проверяем аномалии(выбросы) по правилу IQR"""
    values = (
        pd.Series(
            pd.to_numeric(series, errors="coerce"),
            index=series.index,
            dtype="float64",
        ).dropna()
    )  # приводим в числа, некорректные превращаем в пропуски (coerce) и дропаем
    if values.empty:
        raise ValueError

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low = float(q1 - 1.5 * iqr)
    high = float(q3 + 1.5 * iqr)

    outliers = values[(values < low) | (values > high)]
    return outliers, low, high


def top_pairs(cast_series: pd.Series, n: int = 10) -> list[tuple[tuple[str, str], int]]:
    """какие два актёра чаще всего появляются в одном сериале"""
    pairs: Counter[tuple[str, str]] = Counter()

    for cast in cast_series.dropna():
        names = sorted(
            {
                name.strip()
                for name in cast.split(",")
                if name.strip() and name.strip() != "Unknown"
            }
        )
        pairs.update(combinations(names, 2))

    return pairs.most_common(n)
