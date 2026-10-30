"""Only supplied archives are used. No old project data/code are imported."""
import argparse
import csv
import io
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, digest, write_json

def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    for entry in archive.infolist():
        target = (destination / entry.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe archive path")
    archive.extractall(destination)

def docx_paragraphs(content):
    xml = ET.fromstring(zipfile.ZipFile(io.BytesIO(content)).read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return ["".join(x.itertext()) for x in xml.findall(".//w:p", ns)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-zip", required=True)
    ap.add_argument("--image-zip", required=True)
    args = ap.parse_args()
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for source, name in [(args.text_zip, "asap-sas.zip"), (args.image_zip, "handwritten-asap-sas.zip")]:
        dest = raw / name
        if not dest.exists():
            shutil.copy2(source, dest)
        assert digest(source) == digest(dest)
        records.append({"source": source, "copy": str(dest.relative_to(ROOT)), "sha256": digest(dest), "bytes": dest.stat().st_size})
    a = zipfile.ZipFile(raw / "asap-sas.zip")
    safe_extract(a, raw / "asap")
    desc = zipfile.ZipFile(io.BytesIO(a.read("Data_Set_Descriptions.zip")))
    prompts = {}
    for p in (3, 4):
        name = next(n for n in desc.namelist() if f"#{p}--" in n)
        paragraphs = docx_paragraphs(desc.read(name))
        q_index = next(i for i, t in enumerate(paragraphs) if t.startswith("Prompt—"))
        rubric_index = next(i for i, t in enumerate(paragraphs) if t.startswith("Reading for Information Scoring"))
        rubric = "\n".join(paragraphs[rubric_index + 1:])
        (raw / f"prompt{p}_source.txt").write_text("\n".join(paragraphs), encoding="utf-8")
        prompts[str(p)] = {"question": paragraphs[q_index + 1], "rubric": rubric, "min_score": 0, "max_score": 2,
                           "source": f"Data_Set_Descriptions.zip/{name}",
                           "model_rubric": ("2: develop a strong interpretation with specific relevant text evidence; 1: basic understanding with some relevant evidence; 0: limited, illogical or irrelevant understanding. Pandas and koalas are specialists with restricted diets and stable habitats; pythons are adaptable generalists." if p == 3 else "2: thoughtful critical examination with insightful supported judgments; 1: limited examination with plausible supported judgments; 0: superficial, unsupported or irrelevant. Explain invasive species spreading outside native habitats, threatening biodiversity without natural predators, using article evidence.")}
    write_json(ROOT / "configs/prompts.json", prompts)
    all_rows = {}
    source_stats = {}
    # train_rel_2 is the official labeled training release. Other TSVs are audited only.
    for name in ("train.tsv", "train_rel_2.tsv", "public_leaderboard.tsv", "public_leaderboard_rel_2.tsv", "private_leaderboard.tsv"):
        rows = list(csv.DictReader(io.StringIO(a.read(name).decode("utf-8-sig")), delimiter="\t"))
        keep = [r for r in rows if r.get("EssaySet") in ("3", "4")]
        source_stats[name] = {"columns": list(rows[0]), "counts": dict(Counter(r["EssaySet"] for r in keep)),
                              "score1": dict(Counter(r.get("Score1", "missing") for r in keep))}
        if name == "train_rel_2.tsv":
            for r in keep:
                key = (int(r["EssaySet"]), int(r["Id"]))
                assert key not in all_rows
                assert int(r["Score1"]) in (0, 1, 2)
                all_rows[key] = {"id": int(r["Id"]), "prompt": int(r["EssaySet"]), "answer": r["EssayText"],
                                 "score": int(r["Score1"]), "score2": int(r["Score2"]), "source": name}
    solution = {(int(r["essay_set"]), int(r["id"])): r for r in csv.DictReader(io.StringIO(a.read("public_leaderboard_solution.csv").decode("utf-8-sig"))) if r["essay_set"] in ("3", "4")}
    for r in csv.DictReader(io.StringIO(a.read("public_leaderboard.tsv").decode("utf-8-sig")), delimiter="\t"):
        if r["EssaySet"] not in ("3", "4"):
            continue
        key = (int(r["EssaySet"]), int(r["Id"]))
        if key not in solution:
            continue
        assert key not in all_rows
        score = int(solution[key]["essay_score"])
        assert score in (0, 1, 2)
        all_rows[key] = {"id": key[1], "prompt": key[0], "answer": r["EssayText"], "score": score,
                         "score2": None, "source": "public_leaderboard.tsv+public_leaderboard_solution.csv", "official_test": True}
    h = zipfile.ZipFile(raw / "handwritten-asap-sas.zip")
    safe_extract(h, raw / "handwritten")
    references = {}
    for name in h.namelist():
        if "/information/" in name and name.endswith(".txt"):
            for line in h.read(name).decode("utf-8-sig").splitlines():
                if not line.startswith("SAS_"):
                    continue
                fields = line.split("\t")
                if len(fields) != 5:
                    raise ValueError(f"Unexpected metadata: {line}")
                fname, ident, text, pen, status = fields
                p = int(fname.split("_")[1])
                references[(p, int(ident))] = {"reference": text, "reference_status": status, "pen": pen, "reference_source": name}
    images = []
    for name in h.namelist():
        m = re.search(r"SAS_([34])_(\d+)\.png$", name)
        if not m:
            continue
        key = (int(m[1]), int(m[2]))
        r = {"prompt": key[0], "id": key[1], "image": str((raw / "handwritten" / name).relative_to(ROOT)).replace("\\", "/"),
             "score_match": key in all_rows, "reference": None, "reference_status": "missing"}
        r.update(references.get(key, {}))
        if key in all_rows:
            r.update(score=all_rows[key]["score"], original_answer=all_rows[key]["answer"])
        images.append(r)
    # Freeze an OCR preprocessing development subset before OCR/model evaluation.
    # Sample 6809 was inspected to debug line touching; never call it blind OCR test.
    development_ids = sorted(r["id"] for r in images if r["prompt"]==3 and r["reference_status"]=="ok")[:15]
    for r in images:
        r["ocr_partition"] = "development" if r["prompt"]==3 and r["id"] in development_ids else "test"
    write_json(ROOT / "data/processed/ocr_split_manifest.json", {"development_prompt3_ids":development_ids,
               "policy":"First 15 normal-reference Prompt 3 IDs for generic preprocessing development; no OCR-reference or image used to train/tune scoring models; remaining normal references held out for primary CER/WER."})
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, rows in [("answers", list(all_rows.values())), ("images", images)]:
        with (processed / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            import json
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"archives": records, "text_sources": source_stats, "score_target": "Score1; Score2 only measures rater agreement",
               "scoring_range": {"3": [0, 2], "4": [0, 2]}, "image_counts": dict(Counter(r["prompt"] for r in images)),
               "matched_scores": dict(Counter(r["prompt"] for r in images if r["score_match"])),
               "reference_status": dict(Counter(f'{r["prompt"]}:{r["reference_status"]}' for r in images)),
               "unmatched_images": [r["image"] for r in images if not r["score_match"]],
               "noncommercial_dataset_notice": "Preserve handwritten README; non-commercial research only.",
               "reference_policy": "Only supplied information/*.txt is a verified OCR reference. Matched ASAP answers without that metadata are proxy texts, not verified transcriptions."}
    write_json(processed / "audit.json", summary)
    own = ROOT / "data/self_collected"
    own.mkdir(parents=True, exist_ok=True)
    (own / "labels.example.csv").write_text("sample_id,prompt,image,reference,score\n", encoding="utf-8-sig")
    print(__import__("json").dumps({k:v for k,v in summary.items() if k != "unmatched_images"}, ensure_ascii=False, indent=2))
    print("unmatched images:", len(summary["unmatched_images"]))

if __name__ == "__main__":
    main()
