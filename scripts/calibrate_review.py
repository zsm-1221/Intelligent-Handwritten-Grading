"""Run after OCR evaluation. Threshold fitting reads validation predictions only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json, experiment, digest
from src.evaluation.review import select_threshold, routing_metrics

def main():
    output=experiment("review_calibration")
    registry=read_json(ROOT / "models/registry.json")
    config=read_json(ROOT / "configs/review.json")
    policies={}
    for p in ("3","4"):
        item=registry[p]
        val=read_json(ROOT / "experiments" / item["version"] / "validation_predictions.json")
        policy=select_threshold(val["true"],val["pred"],val["confidence"],target_exact=config["target_exact"],
                                target_within_one=config["target_within_one"],minimum=config["minimum_validation_accepted"])
        write_json(output / f"p{p}_validation_curve.json",policy)
        policies[p]={k:v for k,v in policy.items() if k != "curve"}
        item["review"]=policies[p]
        assert digest(ROOT / item["path"] / "model.safetensors")==item["model_sha256"]
    # Persist chosen thresholds before reading test predictions to assess routing.
    write_json(output / "frozen_validation_policies.json",policies)
    write_json(ROOT / "models/registry.json",registry)
    ep=ROOT / read_json(ROOT / "experiments/evaluation_pointer.json")["path"]
    evaluation=read_json(ep / "deployed_metrics.json")
    for p in ("3","4"):
        item=registry[p]
        pred=read_json(ep / (item["version"]+"_test_predictions.json"))
        evaluation[p]["routing"]=routing_metrics(pred["true"],[r["score"] for r in pred["predictions"]],
                                                    [r["confidence"] for r in pred["predictions"]],policies[p]["threshold"])
    write_json(output / "test_routing.json",evaluation)
    write_json(ROOT / "experiments/review_pointer.json",{"path":str(output.relative_to(ROOT))})
    print("REVIEW_CALIBRATION_COMPLETED",policies,flush=True)

if __name__ == "__main__":
    main()
