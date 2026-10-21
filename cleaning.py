from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """читает и проверяет csv"""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("машина не нашла файл по этому пути")

    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"не хватает столбцов ({missing}) для грамотного анализа")

    return df


def drop_duplicate_keys(df: pd.DataFrame) -> pd.DataFrame:
    """убирает сериалы-дубликаты в таблице"""
    return df.drop_duplicates(subset="show_id")


def fix_shifted_rating(df: pd.DataFrame) -> pd.DataFrame:
    """после проверки данных, оказалось, что у 3 фильмов длительность попала в рейтинг - нехорошо"""
    df = df.copy()
    mask = df["rating"].str.contains(
        "min", na=False
    )  # строки, где в rating лежит длительность, nan (пропуски) меняем на false
    df.loc[mask, "duration"] = df.loc[
        mask, "rating"
    ]  # маска одинаковая что для рейтинга что для длительности чтобы было соответствие
    df.loc[mask, "rating"] = (
        np.nan
    )  # все значения из рейтинга перенесели - теперь можно поставить пропуски
    return df


def missing_share(df: pd.DataFrame) -> pd.Series:
    """сколько процентов таблицы заполнено пропусками (nan)"""
    return (df.isna().mean() * 100).sort_values(
        ascending=False
    )  # для каждого столбца складываем тру фолсы делим на колво и переводим в проценты


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """заполняет пропуски значением Unknown"""
    df = df.copy()
    for col in ["director", "cast", "country", "rating"]:
        df[col] = df[col].fillna("Unknown")
    return df


def parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
    """превращает строку ' December 21, 2021' в понятную машине дату"""
    df = df.copy()
    df["date_added"] = pd.to_datetime(
        df["date_added"].str.strip(),  # данные поступили с лишним пробелом
        format="%B %d, %Y",  # как в std::get_time в плюсах (важно %m - номер месяца а %B - его название)
        errors="coerce",  # чтобы не кидало ошибку на NaT заменим
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """добавляет недостающие для машины признаки"""
    df = df.copy()

    df["year_added"] = df["date_added"].dt.year.astype(
        "Int64"
    )  # вытягивает год и если его нет безопасно переводит в nan-friendly Int64
    df["month_added"] = df["date_added"].dt.month.astype(
        "Int64"
    )  # вытягивает месяц и если его нет безопасно переводит в nan-friendly Int64
    df["years_to_netflix"] = (
        df["year_added"] - df["release_year"]
    )  # вдруг фильм вышел в 2010 году, а на Netflix появился в 2021-м

    df["n_genres"] = (
        df["listed_in"].str.split(", ").str.len()
    )  # в скольких жанрах засветился
    df["n_countries"] = (
        df["country"].str.split(", ").str.len()
    )  # в скольких стрнах заснимался
    df.loc[df["country"] == "Unknown", "n_countries"] = (
        np.nan
    )  # предущий шаг посчитает незаполненные страны как 1, нехорошо

    return df


def split_duration(df: pd.DataFrame) -> pd.DataFrame:
    """разделяет длительность фильмов и сериалов"""
    df = df.copy()
    number = pd.Series(
        pd.to_numeric(df["duration"].str.split().str[0]),
        # от '121 min' или "3 Seasons" берем '121' или "3" и переводим в число
    )

    df["duration_min"] = number.where(
        df["type"] == "Movie"
    )  # идем по строке и делаем новый столбец если это фильм то ставим время иначе - пропуск
    df["seasons"] = number.where(
        df["type"] == "TV Show"
    )  # идем по строке и делаем новый столбец если это сериал то ставим время иначе - пропуск
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """приводит данные в подходящий для машины вид"""
    df = drop_duplicate_keys(df)
    df = fix_shifted_rating(df)
    df = parse_date_added(df)
    df = split_duration(df)
    df = fill_missing(df)
    df = add_features(df)
    return df
