"""Functions to analyze the model"""
# =================
# ==== IMPORTS ====
# =================

import numpy as np
import pandas as pd


# ===================
# ==== FUNCTIONS ====
# ===================

def get_soldout_rows(df: pd.DataFrame, pred: np.array, target: np.array) -> pd.DataFrame:
    """Get the rows with soldouts (ventes >= fourni)

    Args:
        df: dataset
        pred: predictions of the model (newspapers delivered)
        target: target (sales)
    Returns:
        df_soldout: dataset of the sold out rows
    """
    df['ventes'] = target
    df['fourni'] = pred
    df_soldout = df[df["ventes"] >= df["fourni"]]
    return df_soldout


def get_unsold_rows(df: pd.DataFrame, pred: np.array, target: np.array) -> pd.DataFrame:
    """Get the rows with unsold newspapers (ventes < fourni)

    Args:
        df: dataset
        pred: predictions of the model (newspapers delivered)
        target: target (sales)
    Returns:
        df_unsold: dataset of the unsold rows
    """
    df['ventes'] = target
    df['fourni'] = pred
    df_unsold = df[df["ventes"] < df["fourni"]]
    return df_unsold