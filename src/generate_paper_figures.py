from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from generate_access_logs import generate_access_log


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_figure_2(zipf_csv_path: pathlib.Path, output_path: pathlib.Path) -> None:
    df = pd.read_csv(zipf_csv_path).sort_values("tuple_zipf_exponent")

    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        df["tuple_zipf_exponent"],
        df["ml_accuracy"],
        marker="o",
        color="#1f4db6",
        linewidth=2.2,
        label="DT accuracy",
    )
    ax.plot(
        df["tuple_zipf_exponent"],
        df["heuristic_accuracy"],
        linestyle="--",
        color="#6b7280",
        linewidth=2.0,
        label="HFA baseline",
    )
    ax.set_title("Figure 2: DT Accuracy vs. Zipf Exponent")
    ax.set_xlabel("Zipf exponent theta")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_workload_d(base_config: Dict[str, Any]) -> pd.DataFrame:
    cfg = copy.deepcopy(base_config)
    cfg["n_tuples"] = 50000
    cfg["n_transactions"] = 375000

    access_log = generate_access_log(cfg)

    split_idx = int(0.7 * len(access_log))
    hist_log = access_log.iloc[:split_idx].copy()
    future_log = access_log.iloc[split_idx:].copy()

    all_tuples = pd.DataFrame({"tuple_id": range(int(cfg["n_tuples"]))})

    hist_group = hist_log.groupby("tuple_id")
    access_freq = hist_group.size().rename("access_freq")

    recent_hist = hist_log.tail(max(int(cfg["window_size"]), 50000)).copy()
    recent_group = recent_hist.groupby("tuple_id")
    recent_unique_columns = recent_group["column_id"].nunique().rename("recent_unique_columns")

    # Mean inter-arrival gap estimated on recent history.
    recent_gap_df = recent_hist.sort_values("timestamp").copy()
    recent_gap_df["gap"] = recent_gap_df.groupby("tuple_id")["timestamp"].diff()
    mean_gap = recent_gap_df.groupby("tuple_id")["gap"].mean().rename("mean_gap")

    features_df = all_tuples.merge(access_freq, on="tuple_id", how="left")
    features_df = features_df.merge(mean_gap, on="tuple_id", how="left")
    features_df = features_df.merge(recent_unique_columns, on="tuple_id", how="left")

    features_df["access_freq"] = features_df["access_freq"].fillna(0.0)
    features_df["recent_unique_columns"] = features_df["recent_unique_columns"].fillna(0.0)
    features_df["mean_gap"] = features_df["mean_gap"].fillna(float(recent_hist["timestamp"].max()) + 1.0)

    future_counts = future_log.groupby("tuple_id").size().reindex(range(int(cfg["n_tuples"])), fill_value=0)
    k_hot = max(1, int(round(float(cfg["tiering"]["hot_capacity_ratio"]) * int(cfg["n_tuples"]))))
    hot_tuple_ids = set(future_counts.sort_values(ascending=False).head(k_hot).index.to_list())

    labels_df = all_tuples.copy()
    labels_df["is_hot"] = labels_df["tuple_id"].isin(hot_tuple_ids).astype(int)

    enriched = pd.DataFrame(
        {
            "tuple_id": features_df["tuple_id"],
            "access_freq": features_df["access_freq"].astype(float),
            "mean_gap": features_df["mean_gap"].astype(float),
            "recent_unique_columns": features_df["recent_unique_columns"].astype(float),
            "is_hot": labels_df["is_hot"].astype(int),
        }
    )
    return enriched


def plot_figure_3_and_4(dataset_df: pd.DataFrame, output_pr: pathlib.Path, output_fi: pathlib.Path) -> None:
    feature_cols = ["access_freq", "mean_gap", "recent_unique_columns"]
    X = dataset_df[feature_cols]
    y = dataset_df["is_hot"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=80,
        class_weight="balanced",
        random_state=42,
    )
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    dt.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    dt_scores = dt.predict_proba(X_test)[:, 1]
    lr_scores = lr.predict_proba(X_test)[:, 1]

    dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_scores)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_scores)

    dt_ap = average_precision_score(y_test, dt_scores)
    lr_ap = average_precision_score(y_test, lr_scores)

    ensure_parent(output_pr)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(dt_recall, dt_precision, color="#1f4db6", linewidth=2.1, label=f"DT (AP={dt_ap:.3f})")
    ax.plot(lr_recall, lr_precision, color="#2e8b57", linewidth=2.1, label=f"LR (AP={lr_ap:.3f})")
    ax.set_title("Figure 3: Precision-Recall Curves on workload_D (50,000 tuples)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pr, dpi=240)
    plt.close(fig)

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": dt.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    ensure_parent(output_fi)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(fi["feature"], fi["importance"], color="#1f4db6")
    ax.invert_yaxis()
    ax.set_title("Figure 4: Decision Tree Feature Importance Rankings")
    ax.set_xlabel("Gini importance")
    fig.tight_layout()
    fig.savefig(output_fi, dpi=240)
    plt.close(fig)

    fi.to_csv("results/tables/figure4_feature_importance_rankings.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2, Figure 3, and Figure 4 artifacts")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    fig2_out = pathlib.Path("results/figures/figure2_dt_accuracy_vs_zipf.png")
    fig3_out = pathlib.Path("results/figures/figure3_pr_curve_workload_d.png")
    fig4_out = pathlib.Path("results/figures/figure4_dt_feature_importance.png")

    plot_figure_2(pathlib.Path("results/tables/zipf_sensitivity_accuracy.csv"), fig2_out)

    workload_d = build_workload_d(config)
    plot_figure_3_and_4(workload_d, fig3_out, fig4_out)

    print("Generated paper figures:")
    print(f"- {fig2_out}")
    print(f"- {fig3_out}")
    print(f"- {fig4_out}")
    print("Generated ranking table: results/tables/figure4_feature_importance_rankings.csv")


if __name__ == "__main__":
    main()
