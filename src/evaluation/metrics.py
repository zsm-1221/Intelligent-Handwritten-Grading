import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

def metrics(true, pred):
    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    if len(true) == 0:
        return {"n": 0, "qwk": None, "mae": None, "rmse": None, "accuracy": None, "within_one": None}
    q = cohen_kappa_score(true, pred, weights="quadratic", labels=[0, 1, 2]) if len(set(true) | set(pred)) > 1 else float("nan")
    return {"n":len(true), "qwk":float(q) if np.isfinite(q) else None,
            "mae":float(mean_absolute_error(true, pred)), "rmse":float(np.sqrt(mean_squared_error(true, pred))),
            "accuracy":float(accuracy_score(true, pred)), "within_one":float((abs(true-pred) <= 1).mean()),
            "confusion_matrix":confusion_matrix(true, pred, labels=[0,1,2]).tolist()}

def round_score(values):
    return np.clip(np.floor(np.asarray(values) + 0.5), 0, 2).astype(int)

def bootstrap_qwk(true, pred, seed=42, repeats=500):
    rng = np.random.default_rng(seed)
    true, pred = np.asarray(true), np.asarray(pred)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(true), len(true))
        value = metrics(true[idx], pred[idx])["qwk"]
        if value is not None:
            values.append(value)
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None
