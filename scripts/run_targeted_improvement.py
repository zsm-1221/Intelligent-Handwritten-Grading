"""Controlled additional ablation for ordinal scores and minority score-2 labels."""
import sys
import copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json, experiment
from src.training.train import train

def main():
    pointer=read_json(ROOT / "models/campaign_pointer.json")["path"]
    previous=read_json(ROOT / pointer / "campaign.json")
    assert previous["status"]=="completed"
    if "prior_campaign" in previous:
        raise SystemExit("当前实验已包含针对性改进。如需重跑，请先用 run_training.py 新建独立实验。")
    out=experiment("targeted_campaign")
    campaign={"campaign":str(out.relative_to(ROOT)),"prior_campaign":pointer,"runs":copy.deepcopy(previous["runs"]),"status":"running"}
    for p in (3,4):
        best=read_json(ROOT / pointer / f"p{p}_selected_before_repeats.json")
        for seed in (42,123,2026):
            cfg=copy.deepcopy(best["config"])
            cfg.update(seed=seed,noise_probability=.035,class_balance=True,ordinal_loss_weight=.3)
            campaign["runs"].append(train(p,best["task"],cfg,"repeat_targeted"))
            write_json(out / "campaign.json",campaign)
    campaign["status"]="completed"
    write_json(out / "campaign.json",campaign)
    write_json(ROOT / "models/campaign_pointer.json",{"path":str(out.relative_to(ROOT))})
    print("TARGETED_CAMPAIGN_COMPLETED",out,flush=True)

if __name__=="__main__":
    main()
