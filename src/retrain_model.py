# src/retrain_model.py
from pathlib import Path
import json, pandas as pd, numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
TMP_DIR   = DATA_DIR / "tmp"
WEIGHTS   = DATA_DIR / "weights.json"
FEEDBACK  = DATA_DIR / "feedback.csv"

FEATURES = [
    "doc", "bm25", "jacc", "fuzzy", "soft", "para", "cover",
    "raw_score", "years_norm"
]

DEFAULT_WEIGHTS = {
    "features": FEATURES,
    "coef": [0.6, 0.4, 0.2, 0.15, 0.1, 0.1, 0.1, 0.5, 0.05],
    "intercept": 0.0,
    "version": 1,
    "trained_at": None,
    "n_samples": 0
}

def _load_feedback() -> pd.DataFrame:
    if not FEEDBACK.exists():
        return pd.DataFrame(columns=["filename", "label"])
    df = pd.read_csv(FEEDBACK)
    m = {"selected": 1, "rejected": 0}
    df["label"] = df["decision"].str.lower().map(m)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["filename", "label"]]

def _load_results_features() -> pd.DataFrame:
    rows = []
    for fp in sorted(TMP_DIR.glob("results_*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in items:
            comp = r.get("components", {}) or {}
            years = float(r.get("years_experience", 0) or 0.0)
            row = {
                "filename": r.get("filename"),
                "doc": float(comp.get("doc", 0.0)),
                "bm25": float(comp.get("bm25", 0.0)),
                "jacc": float(comp.get("jacc", 0.0)),
                "fuzzy": float(comp.get("fuzzy", 0.0)),
                "soft": float(comp.get("soft", 0.0)),
                "para": float(comp.get("para", 0.0)),
                "cover": float(comp.get("cover", 0.0)),
                "raw_score": float(r.get("raw_score", r.get("score", 0.0))),
                "years_norm": min(years / 40.0, 1.0),
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["filename"] + FEATURES)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["filename"])
    df = df.drop_duplicates(subset=["filename"], keep="first")
    return df

def _save_weights(payload: dict):
    payload["version"] = int(payload.get("version", 1))
    payload["trained_at"] = datetime.utcnow().isoformat() + "Z"
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def retrain_model():
    feats_df = _load_results_features()
    fb_df    = _load_feedback()

    if feats_df.empty or fb_df.empty:
        payload = DEFAULT_WEIGHTS.copy()
        payload["trained_at"] = datetime.utcnow().isoformat() + "Z"
        _save_weights(payload)
        return {"status": "no_data", "n_samples": 0, "weights": payload}

    df = feats_df.merge(fb_df, on="filename", how="inner")
    df = df.dropna(subset=["label"])
    if df.shape[0] < 10:
        payload = DEFAULT_WEIGHTS.copy()
        payload["trained_at"] = datetime.utcnow().isoformat() + "Z"
        _save_weights(payload)
        return {"status": "few_samples", "n_samples": int(df.shape[0]), "weights": payload}

    X = df[FEATURES].fillna(0.0).astype("float32").values
    y = df["label"].astype(int).values

    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)

    coef = clf.coef_[0].astype(float).tolist()
    intercept = float(clf.intercept_[0])
    payload = {
        "features": FEATURES,
        "coef": coef,
        "intercept": intercept,
        "version": 1,
        "trained_at": None,
        "n_samples": int(df.shape[0])
    }
    _save_weights(payload)
    return {"status": "ok", "n_samples": int(df.shape[0]), "weights": payload}
