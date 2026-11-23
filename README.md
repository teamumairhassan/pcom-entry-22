# PCOM-01: npm vulnerability prediction service

Predicts, for one npm package, the number of GitHub-advisory vulnerabilities (low, moderate, high,
critical, unknown) in the 90 days after its latest version.

## How it works
1. **Input:** CSV with one row per (non-prerelease) version of one package, July 2025 to June 2026
   (PCOM-01 prediction schema, no `purl`, no targets).
2. **Features** (`features.py`): the time series is reduced to one row per package: monthly downloads / issues /
   pull requests / advisory history, plus aggregates over all versions (size, file count, dependencies, release rhythm).
   Because nothing depends on one specific version row, it does not matter which row is the highest semver.
3. **Model** (`model.py`): average of a Random Forest and Poisson gradient boosting (HistGradientBoosting).
   `unknown` is always 0.0 (it is 0 in all training data).
4. **Output:** one CSV row with the five `target_ghsa_*_count_90d` columns as floats.
   If anything fails, the service falls back to the average rate instead of failing.

## Files
| File | Purpose |
|---|---|
| `main.py` | FastAPI service: `GET /health`, `POST /predict` (course template) |
| `predictor.py` | `run_model(df)` and the download / predict / upload logic |
| `features.py`, `model.py` | feature building and model |
| `train.py` | trains from every CSV in `data/` and writes `model.joblib` |
| `evaluate.py` | cross-validation by package (no package in both train and test) |
| `test_local.py` | local test that imitates the competition server |
| `data/training_data.csv` | training data (PCOM-01 training schema) |

## Run
```bash
pip install -r requirements.txt
python train.py          # creates model.joblib (about 20 s)
python test_local.py     # checks the full download -> predict -> upload round trip
python main.py          # starts the service on port 8000
curl http://localhost:8000/health      # {"health":"up"}
```
More training data: put more CSV files (same columns) into `data/` and run `python train.py` again.
See `DEPLOY.md` for the VM setup.
"# pcom-entry-22" 
