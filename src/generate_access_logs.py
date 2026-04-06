from __future__ import annotations

import pathlib
from typing import Dict, Any

import numpy as np
import pandas as pd


def _zipf_probabilities(n_items: int, exponent: float) -> np.ndarray:
    ranks = np.arange(1, n_items + 1)
    weights = 1.0 / np.power(ranks, exponent)
    return weights / weights.sum()


def generate_access_log(config: Dict[str, Any]) -> pd.DataFrame:
    seed = int(config["random_seed"])
    n_tuples = int(config["n_tuples"])
    n_columns = int(config["n_columns"])
    n_transactions = int(config["n_transactions"])

    rng = np.random.default_rng(seed)

    tuple_probs = _zipf_probabilities(n_tuples, exponent=1.2)
    column_probs = _zipf_probabilities(n_columns, exponent=1.1)

    tuple_ids = rng.choice(np.arange(n_tuples), size=n_transactions, p=tuple_probs)
    column_ids = rng.choice(np.arange(n_columns), size=n_transactions, p=column_probs)

    access_log = pd.DataFrame(
        {
            "tx_id": np.arange(n_transactions),
            "timestamp": np.arange(n_transactions),
            "tuple_id": tuple_ids,
            "column_id": column_ids,
        }
    )
    return access_log


def save_access_log(df: pd.DataFrame, output_path: str) -> None:
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
