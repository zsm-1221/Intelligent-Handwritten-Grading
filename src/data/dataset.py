import hashlib
import json
import re
import unicodedata
import pandas as pd
from src.common import ROOT, read_json

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\W+", " ", text).strip()

def group_id(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()

def load_split(prompt, split):
    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    return pd.read_json(ROOT / f"data/processed/prompt{prompt}/{split}.jsonl", lines=True)

def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

def context(prompt):
    info = read_json(ROOT / "configs/prompts.json")[str(prompt)]
    return "Question: " + info["question"] + " Rubric: " + info["model_rubric"]

def encode_answers(tokenizer, answers, prompt, max_length):
    # Fixed question/rubric budget preserves an explicit minimum answer budget.
    # The same rule is used at train/eval/serve, including when max_length is tuned.
    ctx_ids = tokenizer.encode(context(prompt), add_special_tokens=False)[:max_length // 2]
    ctx = tokenizer.decode(ctx_ids)
    return tokenizer([ctx] * len(answers), [str(x) for x in answers], truncation="only_second",
                     max_length=max_length, padding="max_length", return_tensors="pt")
