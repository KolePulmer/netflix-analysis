import numpy as np
import pandas as pd
import pytest

import cleaning


def test_load_data_missing_file(tmp_path):
    path = tmp_path / "no_such_file.csv"

    with pytest.raises(FileNotFoundError):
        cleaning.load_data(path)


def test_load_data_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"show_id": ["s1"], "title": ["T"]}).to_csv(path, index=False)

    with pytest.raises(ValueError):
        cleaning.load_data(path)


def test_fix_shifted_rating_moves_duration():
    df = pd.DataFrame(
        {
            "rating": ["74 min", "TV-MA"],
            "duration": [np.nan, "2 Seasons"],
        }
    )

    out = cleaning.fix_shifted_rating(df)

    assert out.loc[0, "duration"] == "74 min"
    assert pd.isna(out.loc[0, "rating"])
    assert out.loc[1, "rating"] == "TV-MA"


def test_parse_date_added_handles_spaces_and_missing_values():
    df = pd.DataFrame(
        {
            "date_added": [
                " August 14, 2020",
                "September 25, 2021",
                np.nan,
            ]
        }
    )

    out = cleaning.parse_date_added(df)

    assert out.loc[0, "date_added"].year == 2020
    assert out.loc[1, "date_added"].month == 9
    assert pd.isna(out.loc[2, "date_added"])


def test_split_duration_separates_movies_and_seasons():
    df = pd.DataFrame(
        {
            "type": ["Movie", "TV Show", "TV Show"],
            "duration": ["90 min", "2 Seasons", "1 Season"],
        }
    )

    out = cleaning.split_duration(df)

    assert out.loc[0, "duration_min"] == 90
    assert out.loc[1, "seasons"] == 2
    assert out.loc[2, "seasons"] == 1
    assert pd.isna(out.loc[0, "seasons"])


def test_unknown_country_gives_nan_n_countries():
    df = pd.DataFrame(
        {
            "date_added": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "release_year": [2019, 2019],
            "listed_in": ["Dramas", "Dramas"],
            "country": ["India, Japan", "Unknown"],
        }
    )

    out = cleaning.add_features(df)

    assert out.loc[0, "n_countries"] == 2
    assert pd.isna(out.loc[1, "n_countries"])


def test_clean_pipeline_end_to_end():
    raw = pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s2"],
            "type": ["Movie", "TV Show", "TV Show"],
            "title": ["M", "T", "T"],
            "director": [np.nan, "X", "X"],
            "cast": ["A, B", np.nan, np.nan],
            "country": ["United States, India", "India", "India"],
            "date_added": [
                " May 1, 2019",
                "June 2, 2020",
                "June 2, 2020",
            ],
            "release_year": [2015, 2020, 2020],
            "rating": ["PG", "TV-MA", "TV-MA"],
            "duration": ["100 min", "3 Seasons", "3 Seasons"],
            "listed_in": ["Dramas, Comedies", "TV Dramas", "TV Dramas"],
            "description": ["d", "d", "d"],
        }
    )

    out = cleaning.clean(raw)

    assert len(out) == 2
    assert out.loc[0, "director"] == "Unknown"
    assert out.loc[0, "years_to_netflix"] == 4
    assert out.loc[0, "n_countries"] == 2
    assert out.loc[1, "n_countries"] == 1
    assert out.loc[0, "n_genres"] == 2
    assert out.loc[1, "seasons"] == 3
