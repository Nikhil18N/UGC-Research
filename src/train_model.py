from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from baseline_hfa import predict_hot_cold_heuristic


FEATURE_COLUMNS = [
    "recent_access_count",
    "recent_unique_columns",
    "total_access_count",
    "recency",
]


def _build_model(model_cfg: Dict[str, Any], seed: int):
    model_type = str(model_cfg["type"]).strip().lower()

    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=600,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    return DecisionTreeClassifier(
        max_depth=int(model_cfg.get("max_depth", 6)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 50)),
        class_weight="balanced",
        random_state=seed,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
    ax.set_title("ML Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_feature_importance(model, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    else:
        return

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": scores,
        }
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#2f6db4")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_and_evaluate(
    config: Dict[str, Any],
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    seed = int(config["random_seed"])

    labels_series = labels_df.set_index("tuple_id").loc[features_df["tuple_id"], "is_hot"]

    X = features_df[FEATURE_COLUMNS]
    y = labels_series.to_numpy()

    test_size = float(config["split"]["test_size"])
    val_size = float(config["split"]["val_size"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    val_ratio_from_train = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio_from_train,
        random_state=seed,
        stratify=y_train_full,
    )

    model = _build_model(config["model"], seed=seed)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    heuristic_cfg = config["heuristic"]
    y_pred_heuristic = predict_hot_cold_heuristic(
        X_test,
        tuple_access_threshold=int(heuristic_cfg["tuple_access_threshold"]),
        column_access_threshold=int(heuristic_cfg["column_access_threshold"]),
    ).to_numpy()

    rows: List[Dict[str, Any]] = []

    val_metrics = _compute_metrics(y_val, y_pred_val)
    test_metrics = _compute_metrics(y_test, y_pred_test)
    rows.append(
        {
            "method": "ml_classifier",
            "split": "validation",
            **val_metrics,
        }
    )
    rows.append(
        {
            "method": "ml_classifier",
            "split": "test",
            **test_metrics,
        }
    )

    baseline_metrics = _compute_metrics(y_test, y_pred_heuristic)
    rows.append(
        {
            "method": "heuristic_hfa_like",
            "split": "test",
            **baseline_metrics,
        }
    )

    metrics_df = pd.DataFrame(rows)

    model_path = pathlib.Path(config["paths"]["model"])
    metrics_path = pathlib.Path(config["paths"]["metrics"])
    confusion_matrix_path = pathlib.Path("results/figures/confusion_matrix.png")
    feature_importance_path = pathlib.Path("results/figures/feature_importance.png")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_df.to_csv(metrics_path, index=False)

    _save_confusion_matrix(y_test, y_pred_test, confusion_matrix_path)
    _save_feature_importance(model, feature_importance_path)

    return metrics_df
