## Workload Model
A synthetic access log is generated with skewed tuple and column popularity to mimic hot spots in transactional workloads.

## Feature Engineering
Per tuple, we compute:
1. Recent access count.
2. Recent unique columns touched.
3. Total access count.
4. Recency (distance from last access).

## Labeling Strategy
Tuples in the top configured fraction by total access frequency are labeled hot, while others are labeled cold.

## Models
Two lightweight options are evaluated:
1. Decision Tree.
2. Logistic Regression.

## Baseline
An HFA-like heuristic baseline predicts hot if both tuple and column thresholds are exceeded.

## Evaluation Metrics
Accuracy, precision, recall, and F1 score are reported. Additional artifacts include confusion matrix and feature importance.
