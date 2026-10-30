"""Validation-selected conservative routing with explicit empirical coverage."""
import numpy as np
from src.evaluation.metrics import metrics

def select_threshold(true, pred, confidence, target_exact=.85, target_within_one=.98, minimum=30):
    true, pred, confidence = map(np.asarray, (true, pred, confidence))
    candidates = []
    for threshold in sorted(set(confidence.tolist())):
        accepted = confidence >= threshold
        if accepted.sum() < minimum:
            continue
        result = metrics(true[accepted], pred[accepted])
        candidates.append({"threshold":float(threshold), "coverage":float(accepted.mean()), **result})
    feasible = [x for x in candidates if x["accuracy"] >= target_exact and x["within_one"] >= target_within_one]
    best = max(feasible, key=lambda x:x["coverage"]) if feasible else None
    return {"threshold":best["threshold"] if best else 1.000001,
            "target_exact":target_exact, "target_within_one":target_within_one, "minimum_validation_accepted":minimum,
            "target_met":best is not None, "validation_accepted":best,
            "policy":"Validation empirical target; no correctness guarantee. If no threshold qualifies, review all.",
            "confidence_meaning":"Model ranking signal, not predicted probability of grading correctness.", "curve":candidates}

def routing_metrics(true, pred, confidence, threshold):
    true, pred, confidence = map(np.asarray, (true,pred,confidence))
    accept = confidence >= threshold
    return {"n":len(true), "automatic_coverage":float(accept.mean()), "review_coverage":float((~accept).mean()),
            "automatic":metrics(true[accept],pred[accept]), "reviewed":metrics(true[~accept],pred[~accept])}
