## Quantitative Results
Primary benchmark (`results/tables/metrics_summary.csv`) shows clear separation between ML and the heuristic baseline on the test split:

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ML classifier | 0.9415 | 1.0000 | 0.8050 | 0.8920 |
| Heuristic HFA-like | 0.7025 | 1.0000 | 0.0083 | 0.0165 |

The main gain comes from recall and F1, indicating that ML captures many more truly hot tuples without sacrificing precision in this workload.

## Effect of Zipf Exponent on Accuracy
Tuple-skew sensitivity (`results/tables/zipf_sensitivity_accuracy.csv`) is summarized below.

| Tuple Zipf Exponent | ML Accuracy | Heuristic Accuracy | Delta (ML - Heuristic) |
|---:|---:|---:|---:|
| 0.8 | 0.9483 | 0.7023 | 0.2460 |
| 1.0 | 0.9650 | 0.7023 | 0.2628 |
| 1.2 | 0.9415 | 0.7025 | 0.2390 |
| 1.4 | 0.9360 | 0.7020 | 0.2340 |
| 1.6 | 0.8000 | 0.7015 | 0.0985 |

Observed trend:
1. ML outperforms the heuristic at every exponent.
2. The margin is largest in moderate skew (1.0-1.2).
3. At very high skew (1.6), ML performance drops but remains superior to the baseline.

Trend plot: `results/figures/zipf_accuracy_trend.png`.

### Figure Captions
Fig. 2. DT classification accuracy vs. Zipf skew exponent theta for the 20,000-tuple workload. HFA heuristic baseline shown as a dashed line. (`results/figures/figure2_dt_accuracy_vs_zipf.png`)

Fig. 3. Precision-Recall curves for DT (blue) and LR (green) on workload_D (50,000 tuples). (`results/figures/figure3_pr_curve_workload_d.png`)

Fig. 4. Gini-based feature importances from the trained Decision Tree; access_freq and mean_gap are the most discriminative features. (`results/figures/figure4_dt_feature_importance.png`)

## Feature Insights
`results/figures/feature_importance.png` indicates that recent access intensity and recency dominate predictive value, which is consistent with hotness intuition in in-memory access traces.

## Error Analysis
`results/figures/confusion_matrix.png` shows reduced false-cold errors relative to the heuristic behavior, aligning with the large recall improvement.

## Systems Perspective
Higher recall for hot tuples improves the practical utility of memory tiering decisions, because fewer genuinely hot tuples are misrouted to cold storage. In an IMDB setting, this is expected to improve effective memory use and reduce avoidable lookup penalties.
