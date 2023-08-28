"""Functions for feature engineering"""
# =================
# ==== IMPORTS ====
# =================

import datetime
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit


# ===================
# ==== FUNCTIONS ====
# ===================


def get_weekend(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Get if the day is weekend or not
    - 0 if the day is a weekday
    - 1 if the day is weekend

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df: dataset with the feature weekend or not
    """
    df.loc[:, f"{feat_date}_weekend"] = 0
    df.loc[df[f"{feat_date}_weekday"] >= 5, f"{feat_date}_weekend"] = 1
    return df


def extract_date_features(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Create features from a datetime feature

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns
        df: dataset with features extracted from the date
    """
    df[feat_date] = pd.to_datetime(df[feat_date], format='%Y%m%d')
    df[f'{feat_date}_year'] = df[feat_date].dt.year
    df[f'{feat_date}_month'] = df[feat_date].dt.month
    df[f'{feat_date}_day'] = df[feat_date].dt.day
    df[f'{feat_date}_weekday'] = df[feat_date].dt.weekday
    df = get_weekend(df, feat_date=feat_date)
    return df


def add_lockdown_periods(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Add lockdown periods to the dataset.
    Each lockdown is identified by a specific int. 0 means no lockdown period.

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df: dataset with the lockdown indicator feature
    """
    df.loc[:, "lockdown"] = 0
    # First lockdown
    df.loc[(df[feat_date] >= datetime.datetime(2020, 3, 17)) & (df[feat_date] <= datetime.datetime(2020, 5, 10)), "lockdown"] = 1
    # Second lockdown
    df.loc[(df[feat_date] >= datetime.datetime(2020, 10, 30)) & (df[feat_date] <= datetime.datetime(2020, 12, 14)), "lockdown"] = 2
    # Third lockdown
    df.loc[(df[feat_date] >= datetime.datetime(2021, 4, 3)) & (df[feat_date] <= datetime.datetime(2021, 5, 2)), "lockdown"] = 3
    return df


def get_split_train_val_cv(
    df: pd.DataFrame, target: pd.Series, n_splits: int
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Split the time serie dataset for cross validation using expanding window

    Args:
        df: dataset
        target: target of the dataset
        n_splits: number of splits to create
    """
    list_train_valid = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, valid_index in tscv.split(df):
        list_train_valid.append((df.loc[train_index], df.loc[valid_index], target.loc[train_index], target.loc[valid_index]))
    return list_train_valid


def get_split_cv_by_week(
    df: pd.DataFrame, df_date: pd.Series, feat_date: str, target: pd.Series, n_splits: int
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Split the time serie dataset for cross validation using expanding window.
    One window correspond to one week.

    Args:
        df: dataset
        target: target of the dataset
        n_splits: number of splits to create
    """
    # Prepare
    list_train_valid = []
    # Get the end date
    df[feat_date] = df_date
    df = df.sort_values(by=feat_date)
    max_date = df[feat_date].max()
    # Create a data range, split by week
    date_range = pd.date_range(end=max_date, freq="W", periods=n_splits+1)
    for i in range(1, len(date_range)):
        train = df[df[feat_date] <= date_range[i-1]].drop(columns=[feat_date])
        if i == len(date_range) - 1:
            valid = df[(df[feat_date] > date_range[i-1]) & (df[feat_date] <= date_range[i])].drop(columns=[feat_date])
        else:
            valid = df[(df[feat_date] > date_range[i-1]) & (df[feat_date] <= date_range[i])].drop(columns=[feat_date])
        target_train = target.loc[train.index]
        target_valid = target.loc[valid.index]
        list_train_valid.append((train, valid, target_train, target_valid))
    return list_train_valid


def add_holidays_period(df: pd.DataFrame, feat_date: str, zone: str="Zone A") -> pd.DataFrame:
    """Add the holidays periods to the dataset

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df: dataset with the holidays indicator
    """
    # Options
    zone_name = zone.replace(" ", "")
    # Load holiday dataset
    df_holidays = pd.read_pickle("../data/processed/holidays_france.pkl")
    df_holidays_zone = df_holidays[df_holidays["Zones"] == zone]
    # Merge closest holiday date
    merged_df = pd.merge_asof(
        df, df_holidays_zone[["date_begin", "date_end", "Description"]], left_on=feat_date, right_on='date_begin', direction='backward'
    )
    # Set the right index as they are lost during merge asof
    merged_df.index = df.index
    # Filter out rows where the Date is before the begining or after the 'end' date
    merged_df = merged_df[(merged_df[feat_date] >= merged_df['date_begin']) & (merged_df[feat_date] <= merged_df['date_end'])]
    merged_df.drop(columns=["date_begin", "date_end"], inplace=True)
    merged_df.rename(columns={"Description": f"Description_{zone_name}"}, inplace=True)
    # Select rows without holidays
    df_no_holidays = df.drop(index=merged_df.index)
    df_no_holidays.loc[:, f"Description_{zone_name}"] = "None"
    df_final = pd.concat([df_no_holidays, merged_df]).sort_values(by=feat_date)
    return df_final
