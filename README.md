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

## Key Output Artifacts
- `results/tables/metrics_summary.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/feature_importance.png`
- `results/figures/hot_cold_distribution.png`

## Suggested Paper Claim
A lightweight supervised model can provide better hot/cold prediction than static heuristics while maintaining low CPU overhead for online tiering decisions in an in-memory DB setting.
