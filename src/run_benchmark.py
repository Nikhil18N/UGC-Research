from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from build_features import build_features_and_labels, save_dataframe
from generate_access_logs import generate_access_log, save_access_log
from train_model import train_and_evaluate


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_hot_cold_distribution(labels_df: pd.DataFrame, output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    counts = labels_df["is_hot"].value_counts().sort_index()
    labels = ["Cold (0)", "Hot (1)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#8a8f98", "#d65f5f"])
    ax.set_title("Hot vs Cold Tuple Distribution")
    ax.set_ylabel("Number of Tuples")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-guided hot/cold benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    access_log = generate_access_log(config)
    save_access_log(access_log, config["paths"]["raw_access_log"])

    features_df, labels_df = build_features_and_labels(access_log, config)
    save_dataframe(features_df, config["paths"]["features"])
    save_dataframe(labels_df, config["paths"]["labels"])

    metrics_df = train_and_evaluate(config, features_df, labels_df)

    save_hot_cold_distribution(
        labels_df,
        config["paths"].get("hot_cold_distribution", "results/figures/hot_cold_distribution.png"),
    )

    print("Benchmark completed.")
    print("Metrics summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
