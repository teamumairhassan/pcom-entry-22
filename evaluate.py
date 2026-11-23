"""Honest accuracy check: 5-fold cross-validation (2 repeats) where every package is either in
training or in testing, never both.  Run:  python evaluate.py"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold

import model as M
from train import build_xy, load_training_data

df = load_training_data()
X, Y = build_xy(df)
yb = Y.sum(1) > 0
print(f"{len(X)} packages, {X.shape[1]} features, {int(yb.sum())} with a vulnerability ({yb.mean() * 100:.1f}%)")

P = np.zeros((2,) + Y.shape)
for k, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=2, random_state=1).split(X)):
    models = M.fit(X.iloc[tr], Y[tr], n_jobs=2, n_estimators=200)
    P[k // 5][te] = M.predict(models, X.iloc[te])
P = P.mean(0)

base = np.tile(Y.mean(0), (len(Y), 1))
res = {"packages": len(X), "vulnerable_packages": int(yb.sum()),
       "mae_model": float(mean_absolute_error(Y, P)), "mae_baseline_mean": float(mean_absolute_error(Y, base)),
       "r2_total_count": float(r2_score(Y.sum(1), P.sum(1))),
       "r2_per_severity": {s: float(r2_score(Y[:, j], P[:, j])) for j, s in enumerate(["low", "moderate", "high", "critical"])},
       "auc_any_vulnerability": float(roc_auc_score(yb, P.sum(1))),
       "avg_precision_any": float(average_precision_score(yb, P.sum(1)))}
print(json.dumps(res, indent=2))
json.dump(res, open("metrics.json", "w"), indent=2)
