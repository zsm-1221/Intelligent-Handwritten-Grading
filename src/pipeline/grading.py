import re
import time
import threading
from functools import lru_cache
from pathlib import Path
import pandas as pd
from src.common import ROOT, read_json, prompt_info
from src.models.scorer import Scorer

_lock = threading.RLock()

@lru_cache(maxsize=2)
def load_scorer(prompt):
    prompt_info(prompt)
    registry = read_json(ROOT / "models/registry.json")
    item = registry[str(prompt)]
    return Scorer(item["path"], prompt), item

@lru_cache(maxsize=1)
def load_ocr():
    from src.ocr.recognizer import HandwritingOCR
    return HandwritingOCR()

def grade_text(prompt, text, ocr=None):
    start = time.perf_counter()
    info = prompt_info(prompt)
    text = str(text or "").strip()
    if not text:
        raise ValueError("请输入英文答案，或先上传图片进行识别。")
    if len(text) > 20000:
        raise ValueError("答案过长，请教师分段复核。")
    with _lock:
        scorer, item = load_scorer(int(prompt))
        prediction = scorer.predict([text])[0]
    reasons = []
    if prediction["confidence"] < item["review"]["threshold"]:
        reasons.append("评分可靠性信号未达到验证集复核阈值")
    if prediction["truncated"]:
        reasons.append("答案超过模型长度限制，部分内容未参与评分")
    if len(re.findall(r"[A-Za-z]", text)) / max(1,len(re.sub(r"\s", "",text))) < .5:
        reasons.append("答案不是清晰的英文文本")
    if len(text.split()) < 3:
        reasons.append("答案过短，建议教师核对")
    if ocr:
        reasons.extend(ocr.get("warnings", []))
        # OCR references exist only for one supplied prompt. Text-model routing
        # is not evidence that handwriting/image-domain grading is reliable.
        reasons.append("手写图片评分的可靠性尚需教师核对识别文本")
    return {"prompt":int(prompt), "answer":text, **prediction, "max_score":info["max_score"],
            "level":{0:"理解不足",1:"部分达标",2:"充分达标"}[prediction["score"]],
            "needs_review":bool(reasons), "review_reasons":reasons, "version":item["version"],
            "review_threshold":item["review"]["threshold"], "seconds":time.perf_counter()-start,
            "confidence_note":"模型排序信号，不代表预测正确率", "ocr":ocr}

def grade_image(prompt, image):
    start = time.perf_counter()
    prompt_info(prompt)
    if image is None:
        raise ValueError("请先上传英文手写答案图片。")
    ocr = load_ocr().recognize(image)
    if not ocr["text"].strip():
        raise ValueError("未识别到有效文字，请重新拍摄或在文本框手动输入。")
    result = grade_text(prompt, ocr["text"], ocr)
    result["seconds"] = time.perf_counter()-start
    return result

def result_row(result, name):
    return {"答案编号":name, "题目":result["prompt"], "识别或输入文本":result["answer"], "预测分数":result["score"],
            "满分":result["max_score"], "评分等级":result["level"], "可靠性信号":round(result["confidence"],4),
            "建议教师复核":"是" if result["needs_review"] else "否", "复核原因":"；".join(result["review_reasons"]),
            "模型版本":result["version"], "耗时（秒）":round(result["seconds"],3)}

def grade_batch(prompt, texts="", files=None):
    prompt_info(prompt)
    rows = []
    for i,line in enumerate(str(texts or "").splitlines(),1):
        if line.strip():
            try:
                rows.append(result_row(grade_text(prompt,line),f"文本 {i}"))
            except Exception as exc:
                rows.append({"答案编号":f"文本 {i}", "建议教师复核":"是", "复核原因":str(exc)})
    for file in files or []:
        try:
            rows.append(result_row(grade_image(prompt,file),Path(file).name))
        except Exception as exc:
            rows.append({"答案编号":Path(file).name, "建议教师复核":"是", "复核原因":str(exc)})
    if not rows:
        raise ValueError("请逐行输入答案，或上传一组手写图片。")
    return pd.DataFrame(rows)
