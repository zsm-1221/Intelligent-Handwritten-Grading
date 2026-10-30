import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from src.common import ROOT, write_json, digest
from src.data.dataset import group_id

def main():
    df = pd.read_json(ROOT / "data/processed/answers.jsonl", lines=True)
    df["group"] = df.answer.map(group_id)
    report = {}
    for p in (3, 4):
        frame = df[df.prompt == p].copy()
        test = frame[frame.source.str.startswith("public_")].copy()
        pool = frame[~frame.source.str.startswith("public_")].copy()
        # Preserve official held-out test set; remove duplicate test text from development.
        excluded = pool[pool.group.isin(test.group)]
        pool = pool[~pool.group.isin(test.group)].reset_index(drop=True)
        train_i, val_i = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(pool, pool.score, pool.group))
        parts = {"train": pool.iloc[train_i], "validation": pool.iloc[val_i], "test": test}
        out = ROOT / f"data/processed/prompt{p}"
        out.mkdir(parents=True, exist_ok=True)
        for name, data in parts.items():
            data.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        excluded.to_json(out / "excluded_test_duplicates.jsonl", orient="records", lines=True, force_ascii=False)
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            assert not set(parts[a].id) & set(parts[b].id)
            assert not set(parts[a].group) & set(parts[b].group)
        report[p] = {"counts": {n:len(d) for n,d in parts.items()}, "class_counts": {n:dict(Counter(map(int,d.score))) for n,d in parts.items()},
                     "excluded_test_duplicate_rows":len(excluded), "sha256": {n:digest(out / f"{n}.jsonl") for n in parts},
                     "policy":"Official public test IDs frozen; test-normalized-text duplicates excluded from development; grouped stratified 80/20 development split; seed 42."}
    write_json(ROOT / "data/processed/split_manifest.json", report)
    print(report)

if __name__ == "__main__":
    main()
