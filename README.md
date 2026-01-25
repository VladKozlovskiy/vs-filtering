# VS Filtering

Image similarity metrics evaluation pipeline.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install exact versions (recommended)
pip install -r requirements.lock

# Or install with flexible versions
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install
```

## Run experiments

```bash
# Single experiment
python main.py --config-name cellpose_dist

# All experiments
bash scripts/run_all.sh

# Specific experiment type
bash scripts/run_lpips_dist.sh
bash scripts/run_lpips_class.sh
bash scripts/run_cellpose_dist.sh
bash scripts/run_cellpose_class.sh
```

## Results

Results are saved to `outputs/{name}_{timestamp}/` directory.
