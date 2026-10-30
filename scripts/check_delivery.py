"""Fail closed: no completion receipt until real artifacts and acceptance pass."""
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json, digest, experiment
from src.data.dataset import jsonl, load_split

def main():
    registry=read_json(ROOT / "models/registry.json")
    assert set(registry)=={"3","4"}
    proofs={}
    for p,item in registry.items():
        assert digest(ROOT / item["path"] / "model.safetensors")==item["model_sha256"]
        summary=read_json(ROOT / "experiments" / item["version"] / "summary.json")
        history=read_json(ROOT / "experiments" / item["version"] / "history.json")
        assert summary["optimizer_steps"]>0 and summary["encoder_weight_max_abs_change"]>0
        assert summary["validation_qwk"]==max(h["validation"]["qwk"] for h in history)
        assert item["review"]["threshold"]>0
        proofs[p]={"version":item["version"],"weight_sha256":item["model_sha256"],"optimizer_steps":summary["optimizer_steps"],
                   "best_epoch":summary["best_epoch"],"encoder_weight_max_abs_change":summary["encoder_weight_max_abs_change"]}
        parts={s:load_split(p,s) for s in ("train","validation","test")}
        for a,b in [("train","validation"),("train","test"),("validation","test")]:
            assert not set(parts[a].id)&set(parts[b].id)
            assert not set(parts[a].group)&set(parts[b].group)
    hp=ROOT / read_json(ROOT / "experiments/handwriting_pointer.json")["path"]
    hw=read_json(hp / "metrics.json")
    assert hw["3"]["images"]==185 and hw["4"]["images"]==165
    assert hw["3"]["verified_ocr"]["n"]==160
    assert hw["4"]["verified_ocr"]["cer"] is None
    for kind in ("browser","api"):
        folder=ROOT / read_json(ROOT / f"experiments/{kind}_pointer.json")["path"]
        assert read_json(folder / "results.json")["passed"]
    junit=ET.parse(ROOT / "reports/pytest.xml").getroot()
    suites=list(junit.iter("testsuite"))
    assert sum(int(s.attrib.get("tests",0)) for s in suites)>=9
    assert sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0))+int(s.attrib.get("skipped",0)) for s in suites)==0
    assert (ROOT / "reports/实验报告.md").stat().st_size>1000
    receipt={"status":"completed_with_documented_data_limits", "project":str(ROOT),"model_proofs":proofs,
             "ocr_images_evaluated":350,"verified_ocr_holdout_n":160,"unmatched_image_score_policy":"Excluded, never fabricated",
             "acceptance":"Pytest, real Chrome upload/OCR/grading/edit/batch, and HTTP mixed batch passed",
             "self_collected_final_test":"Interface delivered; no genuine group-collected samples were provided, so no fabricated results.",
             "shutdown_authorization":"User explicitly requested shutdown only after task completion."}
    write_json(ROOT / "reports/completion_receipt.json",receipt)
    print("DELIVERY_VERIFIED",receipt,flush=True)

if __name__=="__main__":
    main()
