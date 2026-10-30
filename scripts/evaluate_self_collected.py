"""Independent final test interface; never called by training or Optuna."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.common import ROOT, write_json, experiment
from src.pipeline.grading import grade_image, grade_text
from src.evaluation.ocr_metrics import error_counts, aggregate_errors
from src.evaluation.metrics import metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",default="data/self_collected/labels.csv")
    args = ap.parse_args()
    path = ROOT / args.labels
    if not path.exists():
        raise SystemExit("尚未提供独立自采标注，请按 labels.example.csv 建立 labels.csv；没有样本时不生成指标。")
    df = pd.read_csv(path).fillna("")
    required = {"sample_id","prompt","image","reference","score"}
    if not required <= set(df.columns) or len(df)==0 or df.sample_id.duplicated().any():
        raise ValueError("标注表须包含唯一编号及完整字段，且至少有一份真实自采答案。")
    folder = experiment("independent_self_collected")
    records = []
    for row in df.to_dict("records"):
        p,score = int(row["prompt"]),int(row["score"])
        if p not in (3,4) or score not in (0,1,2):
            raise ValueError("题号或人工分数无效")
        image = (path.parent / str(row["image"])).resolve()
        if not image.is_relative_to((ROOT / "data/self_collected").resolve()):
            raise ValueError("自采测试图片必须位于 data/self_collected 内")
        result = grade_image(p,image)
        record = {**row,"result":result}
        if str(row["reference"]).strip():
            record["ocr_errors"] = error_counts(row["reference"],result["answer"])
            record["reference_result"] = grade_text(p,row["reference"])
        records.append(record)
    report = {str(p):metrics([r["score"] for r in records if r["prompt"]==p],[r["result"]["score"] for r in records if r["prompt"]==p]) for p in (3,4)}
    report["ocr"] = aggregate_errors([r["ocr_errors"] for r in records if "ocr_errors" in r])
    write_json(folder / "samples.json",records)
    write_json(folder / "metrics.json",report)
    print(folder,report)

if __name__ == "__main__":
    main()
