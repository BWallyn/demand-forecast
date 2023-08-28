"""Prepare data for modelisation"""
# =================
# ==== IMPORTS ====
# =================

import datetime
import os
import pandas as pd

from typing import Tuple


# ===================
# ==== FUNCTIONS ====
# ===================

def set_type(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Set the type of each feature

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df: dataset with the features' types set
    """
    # Convert date
    df[feat_date] = pd.to_datetime(df[feat_date], format="%Y-%m-%d")
    df["id"] = df["id"].astype(str)
    return df


def check_split(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    """
    print("Train:", df_train.origin.unique())
    print("Test:", df_test.origin.unique())
    print("Train:", df_train.observation_time.min(), df_train.observation_time.max())
    print("Test:", df_test.observation_time.min(), df_test.observation_time.max())


def split_train_test(df: pd.DataFrame, feat_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets and save the target in another file

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df_train: train dataset
        df_test: test dataset
        y_train: target of the train dataset
        y_test: target of the test dataset
    """
    min_date = df[feat_date].min()
    max_date = df[feat_date].max()
    # Split into train and test
    train_percent = .75
    time_between = max_date - min_date
    train_cutoff = min_date + train_percent * time_between
    df_train = df[df["Date"] <= train_cutoff]
    df_test = df[df["Date"] > train_cutoff]
    # Extract target
    y_train = df_train["Ventes"]
    y_test = df_test["Ventes"]
    df_train.drop(columns="Ventes", inplace=True)
    df_test.drop(columns="Ventes", inplace=True)
    return df_train, df_test, y_train, y_test


def get_holidays() -> None:
    """Get holidays dataset from data.education.gouv

    - Download the info of the holidays from data.education.gouv
    - Keep only the french metropolitan zones
    - Drop duplicates to keep just one row by zone and holiday period
    - Select only years 2020 and after
    - Save the dataset processed
    """
    path_data = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
    df_holidays = pd.read_csv(path_data, sep=";").sort_values(by="Date de début")
    # Select only metropolitan France dates
    df_holidays = df_holidays[df_holidays["Zones"].isin(["Zone A", "Zone B", "Zone C"])]
    # Drop duplicates between same zones and dates
    df_holidays.drop_duplicates(subset=["Zones", "Date de début"], inplace=True)
    # Change types
    df_holidays["Date de début"] = pd.to_datetime(df_holidays["Date de début"].str[:10], format="%Y-%m-%d")
    df_holidays["Date de fin"] = pd.to_datetime(df_holidays["Date de fin"].str[:10], format="%Y-%m-%d")
    df_holidays.rename(columns={"Date de début": "date_begin", "Date de fin": "date_end"}, inplace=True)
    # df_holidays["Date de début"] = df_holidays["Date de début"].dt.date
    # Select only after 2020
    df_holidays = df_holidays[df_holidays["date_end"] >= datetime.datetime(2020, 1, 1)]
    df_holidays.to_pickle(path="../data/processed/holidays_france.pkl")


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    path_data = 'data/raw'
    path_save = 'data/processed'
    # Load data
    df = pd.read_csv(os.path.join(path_data, "dataset.gz"), sep=";")
    # Run
    df = set_type(df, feat_date="Date")
    df_train, df_test, y_train, y_test = split_train_test(df=df, feat_date="Date")
    # Save train and test
    df_train.to_pickle(os.path.join(path_save, "train.pkl"))
    df_test.to_pickle(os.path.join(path_save, "test.pkl"))
    y_train.to_csv(os.path.join(path_save, "target_train.csv"))
    y_test.to_csv(os.path.join(path_save, "target_test.csv"))