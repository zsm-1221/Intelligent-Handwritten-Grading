"""Additional HTTP/API checks including a mixed text/image batch and invalid prompt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradio_client import Client, handle_file
from src.common import ROOT, experiment, write_json

client=Client("http://127.0.0.1:7860",verbose=False)
folder=experiment("api_acceptance")
results=[]
image=str(ROOT / "data/raw/handwritten/Handwritten ASAP SAS/prompt-4/SAS_4_9890.png")
r=client.predict(4,handle_file(image),api_name="/grade_image")
assert r[0].strip() and "预测分数" in r[1]
results.append({"test":"题目 4 图片 OCR 和评分","result":r})
r=client.predict(4,"Invasive species threaten native animals.",[handle_file(image)],api_name="/grade_batch")
assert len(r["data"])==2
score_index=r["headers"].index("预测分数")
assert all(row[score_index] in (0,1,2) for row in r["data"])
results.append({"test":"文本与图片混合批量评分","result":r})
try:
    client.predict(5,"This question is unsupported.",api_name="/grade_text")
except Exception as exc:
    assert "仅支持" in str(exc)
    results.append({"test":"不支持题目明确拒绝并提示复核","error":str(exc)})
else:
    raise AssertionError("Unsupported prompt was accepted")
write_json(folder / "results.json",{"passed":True,"results":results})
write_json(ROOT / "experiments/api_pointer.json",{"path":str(folder.relative_to(ROOT))})
print("API_ACCEPTANCE_PASSED",folder)
