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
    """
    """
    df['ventes'] = target
    df['fourni'] = pred
    df_soldout = df[df["ventes"] >= df["fourni"]]
    return df_soldout