"""Verify that the answer-budget metadata correction changed no scores/weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json

runs=sorted((ROOT / "experiments").glob("*_frozen_evaluation"))
assert len(runs)>=2
old,new=runs[-2:]
old_metrics,new_metrics=read_json(old / "deployed_metrics.json"),read_json(new / "deployed_metrics.json")
assert old_metrics==new_metrics
assert read_json(old / "selection_frozen.json")==read_json(new / "selection_frozen.json")
changed,compared=0,0
for file in old.glob("*_test_predictions.json"):
    a,b=read_json(file),read_json(new / file.name)
    assert a["ids"]==b["ids"] and a["true"]==b["true"]
    assert [r["score"] for r in a["predictions"]]==[r["score"] for r in b["predictions"]]
    changed+=sum(x["truncated"]!=y["truncated"] for x,y in zip(a["predictions"],b["predictions"]))
    compared+=len(a["predictions"])
result={"old_evaluation":str(old.relative_to(ROOT)),"final_evaluation":str(new.relative_to(ROOT)),
        "scores_compared":compared,"scores_metrics_and_frozen_weights_unchanged":True,"corrected_conservative_truncation_flags":changed,
        "reason":"Compute actual answer token budget using encoded question/rubric length rather than assuming the entire reserved half was consumed. Encoded input, model weights, predicted scores, hyperparameters and selection were unchanged."}
write_json(ROOT / "reports/evaluation_repeat_verification.json",result)
print(result)
