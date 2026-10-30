import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.common import ROOT, prompt_info, read_json
from src.data.dataset import encode_answers
from src.evaluation.metrics import round_score

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(task, config):
    return AutoModelForSequenceClassification.from_pretrained(
        ROOT / config["pretrained"], local_files_only=True, num_labels=3 if task == "classification" else 1,
        seq_classif_dropout=config["dropout"], dropout=config["dropout"], attention_dropout=config["dropout"],
        problem_type="single_label_classification" if task == "classification" else "regression")

def decode_logits(logits, task):
    if task == "classification":
        prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return prob.argmax(axis=1), prob.max(axis=1), prob
    values = logits.float().squeeze(-1).cpu().numpy()
    pred = round_score(values)
    # Ranking signal only; not calibrated correctness probability.
    confidence = np.exp(-2.0 * np.abs(values - pred))
    return pred, confidence, values

class Scorer:
    def __init__(self, path, prompt):
        self.path = PathLike(path)
        self.prompt = int(prompt)
        prompt_info(prompt)
        self.meta = read_json(self.path / "run.json")
        self.config = self.meta["config"]
        self.task = self.meta["task"]
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path, local_files_only=True).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, answers, batch_size=16):
        output = []
        for i in range(0, len(answers), batch_size):
            batch = answers[i:i+batch_size]
            enc = encode_answers(self.tokenizer, batch, self.prompt, self.config["max_length"])
            first_separator = enc["input_ids"][0].tolist().index(self.tokenizer.sep_token_id)
            available = self.config["max_length"] - first_separator - 2
            logits = self.model(**{k:v.to(self.device) for k,v in enc.items()}).logits
            pred, confidence, raw = decode_logits(logits, self.task)
            for j in range(len(batch)):
                full_length = len(self.tokenizer.encode(str(batch[j]), add_special_tokens=False))
                output.append({"score":int(pred[j]), "confidence":float(confidence[j]),
                               "raw":raw[j].tolist(), "truncated": full_length > available})
        return output

def PathLike(path):
    from pathlib import Path
    p = Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else ROOT / p
