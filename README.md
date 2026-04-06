# Hot/Cold Data Classification in In-Memory Databases Using ML

This repository contains a lightweight, reproducible setup for an undergraduate research project on ML-guided hot/cold data classification in in-memory database workloads.

## Research Goal
Replace heuristic-only hybrid filtering (tuple + column access tracking) with a lightweight ML classifier (Decision Tree or Logistic Regression) to improve classification quality and downstream memory/query efficiency.

## Project Workflow
1. Generate synthetic tuple/column access logs.
2. Build tuple-level features from recent access history.
3. Label data and train a lightweight classifier.
4. Compare against a heuristic baseline.
5. Simulate tiering decisions and report metrics.
6. Write paper sections from produced figures/tables.

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/run_benchmark.py --config configs/experiment.yaml
```

## Zipf Sensitivity Analysis
```powershell
python src/run_zipf_sensitivity.py --config configs/experiment.yaml
```

This sweep generates:
- `results/tables/zipf_sensitivity_accuracy.csv`
- `results/figures/zipf_accuracy_trend.png`

## Paper Figure Generation
```powershell
python src/generate_paper_figures.py --config configs/experiment.yaml
```

This command generates:
- `results/figures/figure2_dt_accuracy_vs_zipf.png`
- `results/figures/figure3_pr_curve_workload_d.png`
- `results/figures/figure4_dt_feature_importance.png`
- `results/tables/figure4_feature_importance_rankings.csv`

## Key Output Artifacts
- `results/tables/metrics_summary.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/feature_importance.png`
- `results/figures/hot_cold_distribution.png`
- `results/figures/zipf_accuracy_trend.png`
- `results/figures/figure2_dt_accuracy_vs_zipf.png`
- `results/figures/figure3_pr_curve_workload_d.png`
- `results/figures/figure4_dt_feature_importance.png`

## Suggested Paper Claim
A lightweight supervised model can provide better hot/cold prediction than static heuristics while maintaining low CPU overhead for online tiering decisions in an in-memory DB setting.
