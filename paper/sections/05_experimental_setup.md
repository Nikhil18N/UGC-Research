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

## Train/Validation/Test Split
Data is split into train/validation/test using stratified sampling to preserve hot/cold balance.

## Reproducibility Controls
A fixed random seed is used for workload generation and model training.

## Execution Command
`python src/run_benchmark.py --config configs/experiment.yaml`
