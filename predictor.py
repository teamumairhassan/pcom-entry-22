"""Prediction logic (no web framework in here, so it is easy to test)."""
import io
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests

from features import MODELLED, TARGETS, features_single_package
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
_BUNDLE = None


def get_bundle():
    """Loads the trained model once (cached). Trains it first if model.joblib is missing."""
    global _BUNDLE
    if _BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            print("model.joblib not found -> training now ...", flush=True)
            import train
            train.main()
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """df = input CSV (all version rows of ONE package). Returns ONE row with the 5 target columns."""
    try:
        b = get_bundle()
        df = df.drop(columns=[c for c in ["purl", *TARGETS] if c in df.columns])
        X = features_single_package(df).reindex(columns=b["columns"], fill_value=0.0)
        p = M.predict(b["models"], X)[0]
    except Exception as e:                                           # never fail: fall back to the average
        print("model failed, using base rate:", repr(e), flush=True)
        try:
            p = np.array(get_bundle()["base_rate"], dtype=float)
        except Exception:
            p = np.zeros(len(MODELLED))
    values = {t: float(round(v, 6)) for t, v in zip(MODELLED, p)}
    values["target_ghsa_unknown_count_90d"] = 0.0                    # always 0 in the training data
    out = pd.DataFrame([values], columns=TARGETS)
    assert len(out) == 1 and not out.isna().any().any()
    return out


def start_prediction(download_url: str, upload_url: str):
    t0 = time.time()
    try:
        result = requests.get(download_url, timeout=30)
        result.raise_for_status()
        input_data = pd.read_csv(io.StringIO(result.text), sep=",", decimal=".")

        prediction_data = run_model(input_data)

        assert len(prediction_data) == 1, "Prediction DataFrame must contain exactly 1 row"
        buf = io.StringIO()
        prediction_data.to_csv(buf, sep=",", decimal=".", header=True, index=False)
        files = {"predictions": ("prediction.csv", buf.getvalue(), "text/csv")}

        upload_res = requests.post(upload_url, files=files, timeout=30)
        upload_res.raise_for_status()
        print(f"Prediction {prediction_data.iloc[0].to_dict()} uploaded in {time.time() - t0:.2f}s; "
              f"server said: {upload_res.text[:300]}", flush=True)
    except Exception as e:
        print("Error:", repr(e), flush=True)
