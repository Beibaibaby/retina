# Two-SAC Retinal Microcircuit Model

```bash
# 1. create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. quick sanity plot
python scripts/run_single.py

# 3. Monte-Carlo robustness (50 repeats)
python scripts/run_batch.py -N 50 --out figs/batch50.png
