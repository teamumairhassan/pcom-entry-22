"""Trains the model from every CSV in data/ and saves it to model.joblib.

Run:  python train.py
Add new crawled data by simply putting more CSV files (PCOM-01 training schema) into data/ and re-running.
"""
import glob
import os
import urllib.parse as up
from datetime import datetime, timezone

import joblib
import pandas as pd

import model as M
from features import MODELLED, features_by_package

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")


def load_training_data(data_dir=os.path.join(HERE, "data")) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.csv.gz")))
    if not files:
        raise SystemExit(f"no CSV files found in {data_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("purl")                                  # same version twice -> keep one
    name = df.purl.str.replace(r"^pkg:npm/", "", regex=True).str.rsplit("@", n=1).str[0]
    return df.assign(pkg=name.map(up.unquote))


def build_xy(df: pd.DataFrame):
    X = features_by_package(df)
    Y = df.groupby("pkg", sort=False)[MODELLED].first().loc[X.index].to_numpy(dtype=float)
    return X, Y


def main():
    df = load_training_data()
    X, Y = build_xy(df)
    print(f"training on {len(X)} packages ({len(df)} version rows), {X.shape[1]} features; "
          f"{int((Y.sum(1) > 0).sum())} packages have a vulnerability in the target window")
    models = M.fit(X, Y)
    bundle = {"models": models, "columns": list(X.columns), "base_rate": Y.mean(0).tolist(),
              "n_packages": len(X), "trained_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
