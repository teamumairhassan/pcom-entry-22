"""Model: average of a Random Forest and Poisson gradient boosting (one booster per severity)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def make_rf(n_jobs=-1, n_estimators=300):
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, n_jobs=n_jobs, random_state=0)


def make_hgb():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=150, max_depth=3,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=0)


def fit(X, Y, n_jobs=-1, n_estimators=300):
    """X: DataFrame (packages x features); Y: array (packages x 4 severities)."""
    rf = make_rf(n_jobs, n_estimators).fit(X, Y)
    hgb = [make_hgb().fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return {"rf": rf, "hgb": hgb}


def predict(models, X):
    a = models["rf"].predict(X)
    b = np.column_stack([m.predict(X) for m in models["hgb"]])
    return np.clip((a + b) / 2, 0, None)
