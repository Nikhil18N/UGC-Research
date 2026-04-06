from __future__ import annotations

import pathlib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def build_features_and_labels(
    access_log: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_tuples = int(config["n_tuples"])
    window_size = int(config["window_size"])
    hot_capacity_ratio = float(config["tiering"]["hot_capacity_ratio"])

    recent_log = access_log.tail(window_size)

    recent_group = recent_log.groupby("tuple_id")
    total_group = access_log.groupby("tuple_id")

    recent_access_count = recent_group.size().rename("recent_access_count")
    recent_unique_columns = recent_group["column_id"].nunique().rename("recent_unique_columns")
    recent_last_seen = recent_group["timestamp"].max().rename("last_seen")
    total_access_count = total_group.size().rename("total_access_count")

    all_tuples = pd.DataFrame({"tuple_id": np.arange(n_tuples)})

    features = all_tuples.merge(recent_access_count, on="tuple_id", how="left")
    features = features.merge(recent_unique_columns, on="tuple_id", how="left")
    features = features.merge(recent_last_seen, on="tuple_id", how="left")
    features = features.merge(total_access_count, on="tuple_id", how="left")

    features[["recent_access_count", "recent_unique_columns", "total_access_count"]] = features[
        ["recent_access_count", "recent_unique_columns", "total_access_count"]
    ].fillna(0)

    max_ts = int(access_log["timestamp"].max())
    features["recency"] = (max_ts - features["last_seen"]).fillna(max_ts + 1)
    features = features.drop(columns=["last_seen"])

    total_counts_full = (
        total_access_count.reindex(np.arange(n_tuples), fill_value=0).reset_index(drop=True)
    )
    k_hot = max(1, int(round(hot_capacity_ratio * n_tuples)))
    sorted_tuple_ids = np.argsort(-total_counts_full.to_numpy(), kind="stable")
    hot_tuple_ids = set(sorted_tuple_ids[:k_hot])

    labels = all_tuples.copy()
    labels["is_hot"] = labels["tuple_id"].isin(hot_tuple_ids).astype(int)

    return features, labels


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
