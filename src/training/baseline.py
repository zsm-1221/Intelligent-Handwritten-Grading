"""Validation-only regularization selection; held-out tests are evaluated later."""
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from src.common import ROOT, experiment, write_json
from src.data.dataset import load_split, context
from src.evaluation.metrics import metrics, round_score

def train_baselines():
    folder = experiment("baselines")
    results = []
    for prompt in (3, 4):
        tr, va = load_split(prompt, "train"), load_split(prompt, "validation")
        text = lambda df: (context(prompt) + " Answer: " + df.answer).tolist()
        for task in ("logistic", "ridge"):
            candidates = []
            best_qwk, best_model = -float("inf"), None
            for strength in ([0.5, 2.0, 8.0] if task == "logistic" else [0.1, 1.0, 10.0]):
                estimator = LogisticRegression(C=strength, max_iter=2000, random_state=42) if task == "logistic" else Ridge(alpha=strength)
                model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)), ("estimator", estimator)])
                model.fit(text(tr), tr.score)
                pred = round_score(model.predict(text(va)))
                result = metrics(va.score, pred)
                candidates.append({"strength":strength, "validation":result})
                if len(candidates) == 1 or result["qwk"] > best_qwk:
                    best_qwk, best_model = result["qwk"], model
            dest = folder / f"prompt{prompt}_{task}.joblib"
            joblib.dump(best_model, dest)
            row = {"prompt":prompt, "task":task, "candidates":candidates, "validation_qwk":best_qwk, "model":str(dest.relative_to(ROOT))}
            results.append(row)
            print(row, flush=True)
    write_json(folder / "results.json", results)
    write_json(ROOT / "models/baseline_registry.json", results)
    return folder

if __name__ == "__main__":
    train_baselines()
