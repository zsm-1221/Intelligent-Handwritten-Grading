import numpy as np
import pytest
from src.common import ROOT, read_json, prompt_info, digest
from src.data.dataset import load_split, jsonl
from src.evaluation.metrics import metrics, round_score
from src.evaluation.review import select_threshold
from src.evaluation.ocr_metrics import error_counts
from src.pipeline.grading import grade_text

@pytest.mark.parametrize("prompt",[3,4])
def test_grouped_split_has_no_leakage(prompt):
    parts = {s:load_split(prompt,s) for s in ("train","validation","test")}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not set(parts[a].id) & set(parts[b].id)
        assert not set(parts[a].group) & set(parts[b].group)
    assert all(set(d.score) <= {0,1,2} for d in parts.values())
    assert parts["test"].source.str.startswith("public_").all()
    assert not parts["train"].source.str.startswith("public_").any()

def test_unmatched_images_never_get_scores():
    images = jsonl(ROOT / "data/processed/images.jsonl")
    assert len(images) == 350
    for image in images:
        assert image["score_match"] == ("score" in image)
        if image["score_match"]:
            test = load_split(image["prompt"],"test")
            row = test[test.id == image["id"]]
            assert len(row)==1 and int(row.iloc[0].score)==image["score"]
    assert sum(not r["score_match"] for r in images) == 1

def test_reliability_falls_back_to_review_all():
    policy = select_threshold([0,0,0,2,2,2],[2,2,2,0,0,0],[.9]*6,minimum=2)
    assert policy["threshold"] > 1
    assert not policy["target_met"]

def test_metrics_and_rounding():
    assert metrics([0,1,2],[0,1,2])["qwk"]==1
    assert metrics([0,0],[2,2])["qwk"]==0
    assert metrics([],[])["qwk"] is None
    assert round_score([-.1,.5,1.5,3]).tolist()==[0,1,2,2]
    assert error_counts("abc def","abc xef")["char_errors"]==1

def test_invalid_requests_rejected():
    with pytest.raises(ValueError,match="仅支持"):
        prompt_info(5)
    with pytest.raises(ValueError,match="请输入"):
        grade_text(3,"")

@pytest.mark.parametrize("prompt",[3,4])
def test_real_model_reload_and_provenance(prompt):
    from src.models.scorer import Scorer
    registry = read_json(ROOT / "models/registry.json")[str(prompt)]
    path = ROOT / registry["path"]
    assert digest(path / "model.safetensors") == registry["model_sha256"]
    scorer = Scorer(path,prompt)
    result = scorer.predict(["Pandas eat bamboo and koalas eat eucalyptus. Pythons adapt to many different environments."])[0]
    assert result["score"] in (0,1,2) and 0 <= result["confidence"] <= 1
    version = registry["version"]
    summary = read_json(ROOT / "experiments" / version / "summary.json")
    assert summary["optimizer_steps"]>0 and summary["encoder_weight_max_abs_change"]>0
    assert summary["best_epoch"] <= summary["epochs_ran"]
    validation=load_split(prompt,"validation")
    reloaded=[r["score"] for r in scorer.predict(validation.answer.tolist())]
    saved=read_json(ROOT / "experiments" / version / "validation_predictions.json")
    assert reloaded==saved["pred"], "Reloaded weights must reproduce validation-best predictions"
    assert np.isclose(metrics(validation.score,reloaded)["qwk"],summary["validation_qwk"])

def test_multiline_and_blank_ocr_preprocessing():
    from PIL import Image
    from src.ocr.recognizer import segment_lines
    assert segment_lines(Image.new("RGB",(500,200),"white"))[0] == []
    path = ROOT / "data/raw/handwritten/Handwritten ASAP SAS/prompt-3/SAS_3_6809.png"
    crops,boxes = segment_lines(Image.open(path))
    assert 4 <= len(crops) <= 12
    assert all(a[1] <= b[1] for a,b in zip(boxes,boxes[1:]))
