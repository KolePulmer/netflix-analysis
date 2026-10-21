from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration", "listed_in", "description",
]


def load_data(path: str | Path) -> pd.DataFrame:
  '''читает и проверяет csv'''
  try:
    df = pd.read_csv(path)
  except FileNotFoundError:
    raise FileNotFoundError("машина не нашла файл по этому пути")

  missing = [col for col in COLUMNS if col not in df.columns]
  if missing:
    raise ValueError(f"не хватает столбцов ({missing}) для грамотного анализа")

  return df

def drop_duplicate_keys(df: pd.DataFrame) -> pd.DataFrame:
    '''убирает сериалы-дубликаты в таблице'''
    return df.drop_duplicates(subset="show_id")


def fix_shifted_rating(df: pd.DataFrame) -> pd.DataFrame:
    '''после проверки данных, оказалось, что у 3 фильмов длительность попала в рейтинг - нехорошо'''
    df = df.copy()
    mask = df["rating"].str.contains("min", na=False)  # строки, где в rating лежит длительность, nan (пропуски) меняем на false
    df.loc[mask, "duration"] = df.loc[mask, "rating"]
    df.loc[mask, "rating"] = np.nan
    return df


def missing_share(df: pd.DataFrame) -> pd.Series:
    '''сколько процентов таблицы заполнено пропусками (nan)'''
    return (df.isna().mean() * 100).sort_values(ascending=False)

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    '''заполняет пропуски значением Unknown'''
    df = df.copy()
    for col in ["director", "cast", "country", "rating"]:
        df[col] = df[col].fillna("Unknown")
    return df


def parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
    '''превращает строку ' December 21, 2021' в понятную машине дату'''
    df = df.copy()
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    '''добавляет недостающие для машины признаки'''
    df = df.copy()

    df["year_added"] = df["date_added"].dt.year.astype("Int64")
    df["month_added"] = df["date_added"].dt.month.astype("Int64")
    df["years_to_netflix"] = df["year_added"] - df["release_year"]

    df["n_genres"] = df["listed_in"].str.split(", ").str.len()
    df["n_countries"] = df["country"].str.split(", ").str.len()
    df.loc[df["country"] == "Unknown", "n_countries"] = np.nan

    return df

def split_duration(df: pd.DataFrame) -> pd.DataFrame:
    """разделяет длительность фильмов и сериалов"""
    df = df.copy()
    number = pd.Series(
            pd.to_numeric(df["duration"].str.split().str[0]),
            index=df.index,
        )

    df["duration_min"] = number.where(df["type"] == "Movie")
    df["seasons"] = number.where(df["type"] == "TV Show")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    '''приводит данные в подходящий для машины вид'''
    df = drop_duplicate_keys(df)
    df = fix_shifted_rating(df)
    df = parse_date_added(df)
    df = split_duration(df)
    df = fill_missing(df)
    df = add_features(df)
    return df
