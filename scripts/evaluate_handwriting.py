import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, write_json, experiment
from src.data.dataset import jsonl
from src.ocr.recognizer import HandwritingOCR
from src.pipeline.grading import load_scorer
from src.evaluation.ocr_metrics import error_counts, aggregate_errors
from src.evaluation.metrics import metrics
import numpy as np
from scipy.stats import spearmanr

def paired_effect(rows, text_key):
    if not rows:
        return {"n":0}
    true=np.array([r["score"] for r in rows])
    clean=np.array([r[text_key]["score"] for r in rows])
    recognized=np.array([r["ocr_prediction"]["score"] for r in rows])
    difference=abs(recognized-true)-abs(clean-true)
    result={"n":len(rows),"score_change_rate":float((clean!=recognized).mean()),
            "worsened_absolute_error_rate":float((difference>0).mean()), "improved_absolute_error_rate":float((difference<0).mean()),
            "mean_added_absolute_error":float(difference.mean())}
    if all("errors" in r for r in rows):
        cer=np.array([r["errors"]["char_errors"]/max(1,r["errors"]["reference_chars"]) for r in rows])
        corr=spearmanr(cer,difference).statistic if len(set(difference))>1 and len(set(cer))>1 else None
        result["spearman_cer_vs_added_absolute_error"]=float(corr) if corr is not None and np.isfinite(corr) else None
        result["cer_groups"]={}
        for name,mask in [("cer<=0.10",cer<=.10),("0.10<cer<=0.25",(cer>.10)&(cer<=.25)),("cer>0.25",cer>.25)]:
            result["cer_groups"][name]={"ocr_grading":metrics(true[mask],recognized[mask]),"text_grading":metrics(true[mask],clean[mask])}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=0,help="0 means all supplied handwriting")
    args = ap.parse_args()
    output = experiment("handwriting_evaluation")
    ocr = HandwritingOCR()
    rows = sorted(jsonl(ROOT / "data/processed/images.jsonl"),key=lambda r:(r["prompt"],r["id"]))
    if args.limit:
        rows = rows[:args.limit]
    results = []
    for i,r in enumerate(rows):
        recognized = ocr.recognize(ROOT / r["image"])
        item = {**r,"ocr":recognized}
        if r["reference"] is not None and r["reference_status"] == "ok":
            item["errors"] = error_counts(r["reference"],recognized["text"])
        scorer,_ = load_scorer(r["prompt"])
        if r["score_match"]:
            item["ocr_prediction"] = scorer.predict([recognized["text"]])[0]
            item["original_text_prediction"] = scorer.predict([r["original_answer"]])[0]
            if r["reference"] is not None and r["reference_status"] == "ok":
                item["reference_prediction"] = scorer.predict([r["reference"]])[0]
        results.append(item)
        write_json(output / "samples.json",results)
        print(f"OCR {i+1}/{len(rows)} p{r['prompt']} id={r['id']} lines={len(recognized['lines'])} seconds={recognized['seconds']:.2f}",flush=True)
    report = {}
    for p in (3,4):
        subset = [r for r in results if r["prompt"] == p]
        all_good = [r for r in subset if "errors" in r]
        good = [r for r in all_good if r.get("ocr_partition") == "test"]
        matched = [r for r in subset if r["score_match"]]
        verified = [r for r in matched if "reference_prediction" in r and r.get("ocr_partition") == "test"]
        report[str(p)] = {"images":len(subset),"verified_ocr":aggregate_errors([r["errors"] for r in good]),
                           "verified_ocr_all_diagnostic":aggregate_errors([r["errors"] for r in all_good]),
                           "preprocessing_development_n":sum(r.get("ocr_partition")=="development" for r in subset),
                           "verified_reference_pair":{"n":len(verified),"reference_grading":metrics([r["score"] for r in verified],[r["reference_prediction"]["score"] for r in verified]),
                                                       "ocr_grading":metrics([r["score"] for r in verified],[r["ocr_prediction"]["score"] for r in verified]),
                                                       "paired_effect":paired_effect(verified,"reference_prediction")},
                           "official_original_text_pair":{"note":"Original ASAP text is a proxy for image transcription where verified metadata is absent.",
                              "original_grading":metrics([r["score"] for r in matched],[r["original_text_prediction"]["score"] for r in matched]),
                              "ocr_grading":metrics([r["score"] for r in matched],[r["ocr_prediction"]["score"] for r in matched]),
                              "paired_effect":paired_effect(matched,"original_text_prediction")},
                           "warnings":sum(bool(r["ocr"]["warnings"]) for r in subset)}
    write_json(output / "metrics.json",report)
    write_json(ROOT / "experiments/handwriting_pointer.json",{"path":str(output.relative_to(ROOT))})
    print(report,flush=True)

if __name__ == "__main__":
    main()
