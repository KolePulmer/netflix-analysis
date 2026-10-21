import numpy as np
import pandas as pd

import analysis


def test_explode_column_splits_and_strips():
    df = pd.DataFrame({"title": ["A"], "country": ["India, Japan, "]})
    out = analysis.explode_column(df, "country")
    assert list(out["country"]) == ["India", "Japan"]


def test_iqr_outliers_finds_only_extreme_value():
    series = pd.Series([1, 2, 3, 4, 5, 100])
    outliers, low, high = analysis.iqr_outliers(series)
    assert list(outliers) == [100]
    assert low < 1 and high < 100


def test_top_pairs_counts_pairs_and_skips_unknown():
    cast = pd.Series(["Ann, Bob, Cid", "Ann, Bob", "Unknown", np.nan])
    result = analysis.top_pairs(cast, n=1)
    assert result == [(("Ann", "Bob"), 2)]
