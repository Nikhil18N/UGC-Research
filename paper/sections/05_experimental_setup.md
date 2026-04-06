## Environment
- Language: Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
- Database mode: in-memory simulation with CSV-backed logs for reproducibility

## Dataset Configuration
Default setup in `configs/experiment.yaml`:
- 20,000 tuples
- 20 columns
- 150,000 transactions
- sliding window of 2,000 accesses
- default tuple Zipf exponent: 1.2
- default column Zipf exponent: 1.1

## Train/Validation/Test Split
Data is split into train/validation/test using stratified sampling to preserve hot/cold balance.

## Reproducibility Controls
A fixed random seed is used for workload generation and model training.

## Zipf Sensitivity Protocol
To study workload skew effects, the tuple Zipf exponent is swept over {0.8, 1.0, 1.2, 1.4, 1.6} while keeping other settings fixed.

Execution command:
`python src/run_zipf_sensitivity.py --config configs/experiment.yaml`

Generated artifacts:
- `results/tables/zipf_sensitivity_accuracy.csv`
- `results/figures/zipf_accuracy_trend.png`

## Execution Command
`python src/run_benchmark.py --config configs/experiment.yaml`
