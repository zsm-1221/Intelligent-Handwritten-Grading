"""Ordered audit-independent training driver. All selection uses validation only."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json, experiment
from src.training.train import train
from src.training.optimize import search

def main():
    folder = experiment("training_campaign")
    records = []
    cfg = read_json(ROOT / "configs/training.json")
    state = {"campaign":str(folder.relative_to(ROOT)), "runs":records, "status":"running"}
    def record(result):
        records.append(result)
        write_json(folder / "campaign.json", state)
    # Genuine full-data initial fine-tuning, before hyperparameter optimization.
    for prompt in (3,4):
        for task in ("classification","regression"):
            record(train(prompt, task, cfg, "formal_initial"))
    for prompt in (3,4):
        for task in ("classification","regression"):
            result = search(prompt, task, cfg["search_trials_per_task_prompt"])
            summary = read_json(ROOT / result["path"] / "summary.json")
            summary["selection_source"] = "optuna_best"
            record(summary)
    # Choose task+configuration by validation QWK for each prompt, then repeat.
    for prompt in (3,4):
        best = max([r for r in records if r["prompt"] == prompt], key=lambda r:r["validation_qwk"])
        selected_cfg = copy.deepcopy(best["config"])
        write_json(folder / f"p{prompt}_selected_before_repeats.json", best)
        for seed in cfg["repeat_seeds"]:
            c = copy.deepcopy(selected_cfg)
            c["seed"] = seed
            record(train(prompt, best["task"], c, "repeat_clean"))
            c["noise_probability"] = .035
            record(train(prompt, best["task"], c, "repeat_noise"))
    state["status"] = "completed"
    write_json(folder / "campaign.json", state)
    write_json(ROOT / "models/campaign_pointer.json", {"path":str(folder.relative_to(ROOT))})
    print("CAMPAIGN_COMPLETED", folder, flush=True)

if __name__ == "__main__":
    main()
