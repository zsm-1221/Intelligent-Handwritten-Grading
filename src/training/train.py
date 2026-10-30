"""Independent PyTorch full fine-tuning with validation-best checkpoints."""
import copy
import gc
import json
import random
import time
import platform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.common import ROOT, seed_all, experiment, read_json, write_json, digest
from src.data.dataset import load_split, encode_answers
from src.data.augmentation import corrupt
from src.models.scorer import create_model, device, decode_logits
from src.evaluation.metrics import metrics

class EncodedDataset(Dataset):
    def __init__(self, frame, tokenizer, prompt, cfg, augment=False):
        answers = frame.answer.astype(str).tolist()
        self.noisy_enc = None
        self.rng = random.Random(cfg["seed"]+1009)
        if augment:
            noisy = [corrupt(x, self.rng, cfg["noise_probability"]) for x in answers]
            self.noisy_enc = encode_answers(tokenizer, noisy, prompt, cfg["max_length"])
        scores = frame.score.tolist()
        self.enc = encode_answers(tokenizer, answers, prompt, cfg["max_length"])
        self.scores = torch.tensor(scores, dtype=torch.long)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, i):
        # Equal examples/optimizer budget: replace half of presentations with noise.
        enc = self.noisy_enc if self.noisy_enc is not None and self.rng.random() < .5 else self.enc
        return {**{k:v[i] for k,v in enc.items()}, "labels":self.scores[i]}

@torch.inference_mode()
def validate(model, loader, dev, task):
    model.eval()
    true, predictions, confidences, losses, raw_values = [], [], [], [], []
    for batch in loader:
        target = batch.pop("labels").to(dev)
        logits = model(**{k:v.to(dev) for k,v in batch.items()}).logits
        loss = torch.nn.functional.cross_entropy(logits, target) if task == "classification" else torch.nn.functional.mse_loss(logits.squeeze(-1), target.float())
        p,c,r = decode_logits(logits, task)
        true.extend(target.cpu().tolist())
        predictions.extend(p.tolist())
        confidences.extend(c.tolist())
        raw_values.extend(r.tolist())
        losses.extend([float(loss)] * len(target))
    return {**metrics(true, predictions), "loss":float(np.mean(losses))}, {"true":true, "pred":predictions, "confidence":confidences, "raw":raw_values}

def train(prompt, task, config=None, stage="formal", trial=None, max_steps=None):
    cfg = copy.deepcopy(config or read_json(ROOT / "configs/training.json"))
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    dev = device()
    folder = experiment(f"p{prompt}_{task}_{stage}_s{cfg['seed']}")
    tr, va = load_split(prompt, "train"), load_split(prompt, "validation")
    tok = AutoTokenizer.from_pretrained(ROOT / cfg["pretrained"], local_files_only=True)
    tr_ds = EncodedDataset(tr, tok, prompt, cfg, augment=cfg["noise_probability"] > 0)
    va_ds = EncodedDataset(va, tok, prompt, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, generator=generator, num_workers=0)
    vl = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=0)
    model = create_model(task, cfg).to(dev)
    class_weights = None
    if cfg.get("class_balance", False):
        counts = torch.bincount(torch.tensor(tr.score.tolist()),minlength=3).float()
        class_weights = (counts.sum() / (3 * counts.clamp_min(1))).sqrt()
        class_weights = (class_weights / class_weights.mean()).to(dev)
    # Witness a pretrained transformer attention parameter changing, not just the head.
    witness_name = "distilbert.transformer.layer.5.attention.q_lin.weight"
    witness_before = dict(model.named_parameters())[witness_name].detach().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(dl) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*cfg["warmup_ratio"]), total_steps)
    use_amp = dev.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float32
    run = {"prompt":prompt, "task":task, "stage":stage, "config":cfg, "device":str(dev), "torch":torch.__version__,
           "gpu":torch.cuda.get_device_name(0) if use_amp else None, "python":platform.python_version(),
           "split_manifest_sha256":digest(ROOT / "data/processed/split_manifest.json"), "train_rows":len(tr), "validation_rows":len(va),
           "effective_train_rows":len(tr_ds), "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad), "mixed_precision":str(dtype),
           "class_weights_from_train_only":class_weights.cpu().tolist() if class_weights is not None else None}
    write_json(folder / "run.json", run)
    print("START", folder.name, json.dumps(run), flush=True)
    history = []
    best = -float("inf")
    bad_epochs, steps, start = 0, 0, time.perf_counter()
    best_epoch, grad_norm = 0, None
    pruned = False
    for ep in range(cfg["epochs"]):
        model.train()
        total_loss, seen = 0., 0
        epoch_start = time.perf_counter()
        for batch in dl:
            labels = batch.pop("labels").to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=dtype, enabled=use_amp and dtype != torch.float32):
                logits = model(**{k:v.to(dev) for k,v in batch.items()}).logits
                if task == "classification":
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=cfg["label_smoothing"])
                    if cfg.get("ordinal_loss_weight", 0) > 0:
                        expectation = (torch.softmax(logits.float(),dim=-1) * torch.arange(3,device=dev)).sum(-1)
                        loss = loss + cfg["ordinal_loss_weight"] * torch.nn.functional.mse_loss(expectation,labels.float())
                else:
                    if class_weights is None:
                        loss = torch.nn.functional.mse_loss(logits.squeeze(-1).float(), labels.float())
                    else:
                        per_item = torch.nn.functional.mse_loss(logits.squeeze(-1).float(), labels.float(),reduction="none")
                        loss = (per_item * class_weights[labels]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
            steps += 1
            if steps % 25 == 0:
                print(f"{folder.name} epoch={ep+1} step={steps} loss={total_loss/seen:.4f}", flush=True)
            if max_steps and steps >= max_steps:
                break
        result, predictions = validate(model, vl, dev, task)
        row = {"epoch":ep+1, "train_loss":total_loss/seen, "validation":result, "learning_rate":scheduler.get_last_lr()[0],
               "optimizer_steps":steps, "epoch_seconds":time.perf_counter()-epoch_start, "gradient_norm_last":grad_norm}
        history.append(row)
        write_json(folder / "history.json", history)
        print("EPOCH", folder.name, json.dumps(row), flush=True)
        objective = result["qwk"] if result["qwk"] is not None else -1
        if objective > best:
            best, bad_epochs, best_epoch = objective, 0, ep+1
            checkpoint = folder / "checkpoint"
            model.save_pretrained(checkpoint)
            tok.save_pretrained(checkpoint)
            write_json(checkpoint / "run.json", run)
            write_json(folder / "validation_predictions.json", {"ids":va.id.tolist(), **predictions})
        else:
            bad_epochs += 1
        if trial is not None:
            trial.report(objective, ep)
            if trial.should_prune():
                pruned = True
                break
        if bad_epochs >= cfg["patience"] or (max_steps and steps >= max_steps):
            break
    change = float((dict(model.named_parameters())[witness_name].detach().cpu()-witness_before).abs().max())
    summary = {"path":str(folder.relative_to(ROOT)), "prompt":prompt, "task":task, "stage":stage, "seed":cfg["seed"],
               "validation_qwk":best, "best_epoch":best_epoch, "epochs_ran":len(history), "optimizer_steps":steps,
               "elapsed_seconds":time.perf_counter()-start, "encoder_weight_max_abs_change":change, "witness_parameter":witness_name,
               "last_gradient_norm":grad_norm, "pruned":pruned, "config":cfg}
    assert steps > 0 and change > 0, "No genuine transformer weight update"
    write_json(folder / "summary.json", summary)
    del model, optimizer, scheduler, dl, vl
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    if pruned:
        import optuna
        raise optuna.TrialPruned()
    return summary

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, choices=[3,4], required=True)
    ap.add_argument("--task", choices=["classification","regression"], default="classification")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    print(train(args.prompt, args.task, stage="pilot" if args.pilot else "formal", max_steps=8 if args.pilot else None))
