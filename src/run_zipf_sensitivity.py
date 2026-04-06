from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from build_features import build_features_and_labels, save_dataframe
from generate_access_logs import generate_access_log, save_access_log
from train_model import train_and_evaluate


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _exp_token(exponent: float) -> str:
    return str(exponent).replace(".", "p")


def _apply_paths(cfg: Dict[str, Any], exp: float) -> None:
    token = _exp_token(exp)
    cfg["paths"]["raw_access_log"] = f"data/raw/access_log_zipf_{token}.csv"
    cfg["paths"]["features"] = f"data/processed/features_zipf_{token}.csv"
    cfg["paths"]["labels"] = f"data/processed/labels_zipf_{token}.csv"
    cfg["paths"]["model"] = f"results/model/hot_cold_model_zipf_{token}.joblib"
    cfg["paths"]["metrics"] = f"results/tables/metrics_zipf_{token}.csv"
    cfg["paths"]["confusion_matrix"] = f"results/figures/confusion_matrix_zipf_{token}.png"
    cfg["paths"]["feature_importance"] = f"results/figures/feature_importance_zipf_{token}.png"
    cfg["paths"]["hot_cold_distribution"] = f"results/figures/hot_cold_distribution_zipf_{token}.png"


def run_once(base_config: Dict[str, Any], tuple_zipf_exp: float) -> Dict[str, float]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("zipf", {})
    cfg["zipf"]["tuple_exponent"] = float(tuple_zipf_exp)
    _apply_paths(cfg, tuple_zipf_exp)

    access_log = generate_access_log(cfg)
    save_access_log(access_log, cfg["paths"]["raw_access_log"])

    features_df, labels_df = build_features_and_labels(access_log, cfg)
    save_dataframe(features_df, cfg["paths"]["features"])
    save_dataframe(labels_df, cfg["paths"]["labels"])

    metrics_df = train_and_evaluate(cfg, features_df, labels_df)

    ml_test = metrics_df[(metrics_df["method"] == "ml_classifier") & (metrics_df["split"] == "test")].iloc[0]
    heuristic_test = metrics_df[
        (metrics_df["method"] == "heuristic_hfa_like") & (metrics_df["split"] == "test")
    ].iloc[0]

    return {
        "tuple_zipf_exponent": float(tuple_zipf_exp),
        "ml_accuracy": float(ml_test["accuracy"]),
        "ml_f1": float(ml_test["f1"]),
        "heuristic_accuracy": float(heuristic_test["accuracy"]),
        "heuristic_f1": float(heuristic_test["f1"]),
        "accuracy_delta_ml_minus_heuristic": float(ml_test["accuracy"] - heuristic_test["accuracy"]),
        "f1_delta_ml_minus_heuristic": float(ml_test["f1"] - heuristic_test["f1"]),
    }


def save_plot(results_df: pd.DataFrame, output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        results_df["tuple_zipf_exponent"],
        results_df["ml_accuracy"],
        marker="o",
        linewidth=2,
        label="ML classifier",
    )
    ax.plot(
        results_df["tuple_zipf_exponent"],
        results_df["heuristic_accuracy"],
        marker="s",
        linewidth=2,
        label="Heuristic baseline",
    )
    ax.set_title("Effect of Tuple Zipf Exponent on Test Accuracy")
    ax.set_xlabel("Tuple Zipf Exponent")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zipf exponent sensitivity analysis")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    base_config = load_config(args.config)
    sweep: List[float] = [float(x) for x in base_config.get("zipf", {}).get("sweep_exponents", [1.2])]

    rows = [run_once(base_config, exp) for exp in sweep]
    results_df = pd.DataFrame(rows).sort_values("tuple_zipf_exponent").reset_index(drop=True)

    csv_out = pathlib.Path("results/tables/zipf_sensitivity_accuracy.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_out, index=False)

    save_plot(results_df, "results/figures/zipf_accuracy_trend.png")

    print("Zipf sensitivity analysis completed.")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
