$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment if missing..."
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "Activating environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Running benchmark..."
python src/run_benchmark.py --config configs/experiment.yaml

Write-Host "Done. Check results/tables and results/figures."
