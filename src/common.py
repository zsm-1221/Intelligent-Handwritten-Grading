"""Project-local paths, immutable experiment records and reproducible seeds."""
import json
import hashlib
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def experiment(name):
    path = ROOT / "experiments" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + name)
    path.mkdir(parents=True, exist_ok=False)
    return path

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prompt_info(prompt):
    if str(prompt) not in ("3", "4"):
        raise ValueError("仅支持题目 3 和题目 4，请教师人工复核。")
    return read_json(ROOT / "configs/prompts.json")[str(prompt)]
