"""Turns the time series of version records of ONE package into ONE feature row.

Why one row per package: all version records of a package share the same monthly data and
the same target, so the package (not the version) is the unit we learn from. We still use the
whole time series: static version data (size, file count, dependencies) is aggregated over all
versions, so it does not matter which row is the "highest semver" (the input has no version column).
"""
import numpy as np
import pandas as pd

SEV = ["low", "moderate", "high", "critical", "unknown"]
TARGETS = [f"target_ghsa_{s}_count_90d" for s in SEV]          # output order of the prediction
MODELLED = TARGETS[:4]                                          # "unknown" is always 0 in the data
REF = pd.Timestamp("2026-06-30", tz="UTC")                      # end of the 12-month input period

STATIC = ["unpacked_size", "file_count", "dependency_count", "optional_dependency_count",
          "peer_dependency_count", "dev_dependency_count", "bundled_dependency_count"]
_SKIP_SUBSTR = ("download_days_observed", "unknown")            # constant columns
_SKIP_EXACT = {"purl", "pkg", "published_at", "package_created_at", "_pub", "source"}


def _ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def _val(f, key):
    v = f.get(key, 0.0)
    return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def package_features(g: pd.DataFrame) -> dict:
    """g = all version rows of one package. Returns a flat dict of numeric features."""
    g = g.copy()
    for c in ("published_at", "package_created_at", *STATIC):      # tolerate missing columns
        if c not in g.columns:
            g[c] = np.nan
    g["_pub"] = _ts(g["published_at"])
    g = g.sort_values("_pub", kind="stable")
    last = g.iloc[-1]

    f = {}
    # package-level columns (identical in every row of a package): monthly downloads, issues, PRs, GHSA history
    for c in g.columns:
        if c in _SKIP_EXACT or c in STATIC or c in TARGETS or any(s in c for s in _SKIP_SUBSTR):
            continue
        if pd.api.types.is_numeric_dtype(g[c]):
            f[c] = last[c]

    # version-level columns: aggregate over the whole time series of versions
    for c in STATIC:
        v = pd.to_numeric(g[c], errors="coerce")
        f[f"{c}_last"], f[f"{c}_median"], f[f"{c}_max"] = v.iloc[-1], v.median(), v.max()
    for c in ("unpacked_size", "file_count", "dependency_count"):
        v = pd.to_numeric(g[c], errors="coerce")
        f[f"{c}_growth"] = v.iloc[-1] - v.iloc[0]

    pub = g["_pub"].dropna()
    n = len(g)
    span = (pub.max() - pub.min()).days if len(pub) else 0
    f["n_versions"] = n
    f["span_days"] = span
    f["mean_gap_days"] = span / (n - 1) if n > 1 else 0
    f["days_since_last_publish"] = (REF - pub.max()).days if len(pub) else 0
    f["versions_last_90d"] = int((pub > REF - pd.Timedelta(days=90)).sum())
    created = _ts(last["package_created_at"])
    f["pkg_age_days"] = (REF - created).days if pd.notna(created) else 0
    f["no_repo_data"] = 1.0 if pd.isna(last.get("issues_opened_M0", np.nan)) else 0.0

    # derived features
    f["dl_trend"] = np.log1p(_val(f, "package_downloads_M11")) - np.log1p(_val(f, "package_downloads_M0"))
    for s in SEV[:4]:
        f[f"ghsa_{s}_last6m"] = sum(_val(f, f"ghsa_{s}_count_M{i}") for i in range(6, 12))
    f["ghsa_total_hist"] = sum(_val(f, f"ghsa_{s}_count") for s in SEV[:4])
    f["ghsa_recent12"] = sum(_val(f, f"ghsa_{s}_count_M{i}") for s in SEV[:4] for i in range(12))
    f["issues_opened_12m"] = sum(_val(f, f"issues_opened_M{i}") for i in range(12))
    f["prs_merged_12m"] = sum(_val(f, f"pull_requests_merged_M{i}") for i in range(12))
    f["releases_12m"] = sum(_val(f, f"package_stable_release_count_M{i}") for i in range(12))
    return f


def to_matrix(rows: dict) -> pd.DataFrame:
    X = pd.DataFrame.from_dict(rows, orient="index")
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def features_by_package(df: pd.DataFrame, key: str = "pkg") -> pd.DataFrame:
    """Training: one feature row per package (df has a 'pkg' column)."""
    return to_matrix({name: package_features(g) for name, g in df.groupby(key, sort=False)})


def features_single_package(df: pd.DataFrame) -> pd.DataFrame:
    """Prediction: the whole input CSV is one package."""
    return to_matrix({"input": package_features(df)})
