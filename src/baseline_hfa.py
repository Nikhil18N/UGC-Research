from __future__ import annotations

import pandas as pd


def predict_hot_cold_heuristic(
    features_df: pd.DataFrame,
    tuple_access_threshold: int,
    column_access_threshold: int,
) -> pd.Series:
    """
    Simple HFA-like rule combining tuple and column access signals.
    """
    return (
        (features_df["recent_access_count"] >= tuple_access_threshold)
        & (features_df["recent_unique_columns"] >= column_access_threshold)
    ).astype(int)
