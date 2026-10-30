"""Freeze deployment decision on validation BEFORE opening official test splits."""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src.common import ROOT, read_json, write_json, digest, experiment
from src.data.dataset import load_split, context
from src.evaluation.metrics import metrics, round_score, bootstrap_qwk
from src.evaluation.review import select_threshold, routing_metrics
from src.models.scorer import Scorer

def main():
    campaign_path = ROOT / read_json(ROOT / "models/campaign_pointer.json")["path"]
    campaign = read_json(campaign_path / "campaign.json")
    assert campaign["status"] == "completed"
    output = experiment("frozen_evaluation")
    registry = {}
    stability = {}
    for prompt in (3,4):
        runs = [r for r in campaign["runs"] if r["prompt"] == prompt and r["stage"] in ("repeat_clean","repeat_noise","repeat_targeted")]
        family = {}
        for stage in ("repeat_clean","repeat_noise","repeat_targeted"):
            subset = [r for r in runs if r["stage"] == stage]
            values = [r["validation_qwk"] for r in subset]
            family[stage] = {"mean_validation_qwk":float(np.mean(values)), "std_validation_qwk":float(np.std(values,ddof=1)),
                             "seeds":[r["seed"] for r in subset], "runs":[r["path"] for r in subset]}
        chosen_family = max(family, key=lambda f:family[f]["mean_validation_qwk"])
        # Seed 42 fixed in advance, not cherry-picked using test or seed performance.
        chosen = next(r for r in runs if r["stage"] == chosen_family and r["seed"] == 42)
        version = Path(chosen["path"]).name
        dest = ROOT / f"models/prompt{prompt}" / version
        if dest.exists():
            assert digest(dest / "model.safetensors") == digest(ROOT / chosen["path"] / "checkpoint/model.safetensors"), "Existing deployment must never be overwritten"
        else:
            shutil.copytree(ROOT / chosen["path"] / "checkpoint",dest)
        registry[str(prompt)] = {"path":str(dest.relative_to(ROOT)), "version":version, "task":chosen["task"],
                                 "selection":"Family mean validation QWK; preselected seed 42 checkpoint", "validation_qwk":chosen["validation_qwk"],
                                 "model_sha256":digest(dest / "model.safetensors"), "review":{"status":"pending_validation_calibration"}}
        stability[str(prompt)] = {"families":family,"chosen_family":chosen_family}
    # This frozen file is committed to disk before any held-out test is opened.
    write_json(ROOT / "models/registry.json", registry)
    write_json(output / "selection_frozen.json", registry)
    write_json(output / "seed_stability.json", stability)
    all_runs = []
    base_campaign = Path(campaign.get("prior_campaign",str(campaign_path.relative_to(ROOT))))
    first_timestamp = base_campaign.name.split("_")[0]
    last_timestamp = max(Path(r["path"]).name.split("_")[0] for r in campaign["runs"])
    for path in (ROOT / "experiments").glob("*/summary.json"):
        if not first_timestamp <= path.parent.name.split("_")[0] <= last_timestamp:
            continue
        r = read_json(path)
        if r["stage"] != "pilot" and not r.get("pruned"):
            all_runs.append(r)
    rows, deployed = [], {}
    for prompt in (3,4):
        te = load_split(prompt,"test")
        for r in [r for r in all_runs if r["prompt"] == prompt]:
            scorer = Scorer(ROOT / r["path"] / "checkpoint",prompt)
            pred = scorer.predict(te.answer.tolist())
            score = [x["score"] for x in pred]
            result = metrics(te.score,score)
            rows.append({"prompt":prompt, "method":"distilbert_"+r["task"], "stage":r["stage"], "seed":r["seed"],
                         "validation_qwk":r["validation_qwk"], "run":r["path"], **result})
            write_json(output / (Path(r["path"]).name+"_test_predictions.json"),
                       {"ids":te.id.tolist(),"true":te.score.tolist(),"predictions":pred})
            del scorer
        for r in [r for r in read_json(ROOT / "models/baseline_registry.json") if r["prompt"] == prompt]:
            baseline = joblib.load(ROOT / r["model"])
            values = round_score(baseline.predict((context(prompt)+" Answer: "+te.answer).tolist()))
            rows.append({"prompt":prompt,"method":"tfidf_"+r["task"],"stage":"baseline","seed":42,
                         "validation_qwk":r["validation_qwk"],"run":r["model"],**metrics(te.score,values)})
        scorer = Scorer(registry[str(prompt)]["path"],prompt)
        pred = scorer.predict(te.answer.tolist())
        scores, confidence = [x["score"] for x in pred],[x["confidence"] for x in pred]
        human = load_split(prompt,"validation")
        deployed[str(prompt)] = {"test":metrics(te.score,scores), "test_qwk_bootstrap_95ci":bootstrap_qwk(te.score,scores),
                                 "validation_human_agreement":metrics(human.score,human.score2),
                                 "version":registry[str(prompt)]["version"]}
        err = te.copy()
        err["prediction"],err["confidence"] = scores,confidence
        err["absolute_error"] = abs(err.score-err.prediction)
        err.sort_values("absolute_error",ascending=False).to_csv(output / f"p{prompt}_error_analysis.csv",index=False,encoding="utf-8-sig")
        del scorer
    write_json(output / "comparison.json", rows)
    pd.DataFrame([{k:v for k,v in r.items() if k != "confusion_matrix"} for r in rows]).to_csv(output / "comparison.csv", index=False)
    write_json(output / "deployed_metrics.json", deployed)
    write_json(ROOT / "experiments/evaluation_pointer.json", {"path":str(output.relative_to(ROOT))})
    print("EVALUATION_COMPLETED",output, deployed,flush=True)

if __name__ == "__main__":
    main()
