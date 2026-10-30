"""Actual browser upload/OCR/regrade/text/batch acceptance against running Gradio."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from src.common import ROOT, write_json, experiment

def main():
    output = experiment("browser_acceptance")
    steps, errors = [], []
    def record(name, extra=None):
        steps.append({"test":name,"passed":True,**(extra or {})})
        write_json(output / "results.json",{"steps":steps,"browser_errors":errors})
        print("PASSED",name,flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome",headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1100},locale="zh-CN")
        page.on("pageerror",lambda e:(errors.append(str(e)),print("BROWSER_ERROR",str(e),flush=True)))
        page.goto("http://127.0.0.1:7860",wait_until="domcontentloaded",timeout=60000)
        expect(page.get_by_role("heading",name="英文手写简答题智能评分",exact=True)).to_be_visible(timeout=60000)
        record("中文页面启动")
        page.screenshot(path=str(output / "01_initial.png"),full_page=True)
        image=ROOT / "data/raw/handwritten/Handwritten ASAP SAS/prompt-3/SAS_3_6809.png"
        page.locator('input[type="file"]').first.set_input_files(str(image))
        page.get_by_role("button",name="识别图片并评分",exact=True).click()
        page.wait_for_timeout(5000)
        page.screenshot(path=str(output / "upload_processing.png"),full_page=True)
        print("AFTER_UPLOAD",page.get_by_label("OCR 识别文本（可手动修改）").input_value()[:100],flush=True)
        expect(page.get_by_label("OCR 识别文本（可手动修改）")).not_to_have_value("",timeout=180000)
        expect(page.locator(".score-card").filter(has_text="预测分数").filter(visible=True)).to_be_visible(timeout=180000)
        text=page.get_by_label("OCR 识别文本（可手动修改）").input_value()
        record("图片上传、真实 TrOCR 识别及评分",{"ocr_text":text})
        page.screenshot(path=str(output / "02_image_graded.png"),full_page=True)
        revised="Pandas and koalas are specialists with restricted diets. Pandas eat bamboo and koalas eat eucalyptus leaves. Pythons are generalists and adapt to many habitats."
        page.get_by_label("OCR 识别文本（可手动修改）").fill(revised)
        page.get_by_role("button",name="按修改后的文本重新评分",exact=True).click()
        page.wait_for_timeout(2500)
        expect(page.locator(".score-card").filter(has_text="预测分数").filter(visible=True)).to_be_visible()
        record("人工修改识别文本并重新调用训练模型")
        page.screenshot(path=str(output / "03_regraded.png"),full_page=True)
        page.get_by_role("tab",name="直接输入文本评分",exact=True).click()
        page.get_by_label("英文学生答案",exact=True).fill(revised)
        page.get_by_role("button",name="开始文本评分",exact=True).click()
        expect(page.locator(".score-card").filter(has_text="评分等级").filter(visible=True)).to_be_visible(timeout=60000)
        record("直接输入英文文本评分")
        page.get_by_role("tab",name="批量评分",exact=True).click()
        page.get_by_label("批量英文答案（每行一份）",exact=True).fill(revised+"\nPandas are reptiles and eat the same food as snakes.")
        page.get_by_role("button",name="开始批量评分",exact=True).click()
        expect(page.get_by_text("文本 1",exact=True).filter(visible=True).first).to_be_visible(timeout=60000)
        expect(page.get_by_text("文本 2",exact=True).filter(visible=True).first).to_be_visible(timeout=60000)
        record("批量文本评分")
        page.screenshot(path=str(output / "04_batch.png"),full_page=True)
        page.get_by_label("选择题目",exact=True).fill("题目 4")
        page.get_by_label("选择题目",exact=True).press("ArrowDown")
        page.get_by_role("option").filter(has_text="题目 4：入侵物种").click()
        expect(page.get_by_text("Explain the significance",exact=False)).to_be_visible(timeout=60000)
        record("切换题目 4 并加载对应题目与评分标准")
        page.get_by_role("tab",name="直接输入文本评分",exact=True).click()
        page.get_by_label("英文学生答案",exact=True).fill("Invasive animals spread outside their native habitat and threaten biodiversity. Burmese pythons in the Everglades have no natural predators and eat endangered species.")
        page.get_by_role("button",name="开始文本评分",exact=True).click()
        page.wait_for_timeout(2500)
        expect(page.locator(".score-card").filter(has_text="预测分数").filter(visible=True)).to_be_visible()
        record("题目 4 真实模型评分")
        page.screenshot(path=str(output / "05_prompt4.png"),full_page=True)
        browser.close()
    write_json(output / "results.json",{"steps":steps,"browser_errors":errors,"passed":not errors})
    if errors:
        raise RuntimeError(errors)
    write_json(ROOT / "experiments/browser_pointer.json",{"path":str(output.relative_to(ROOT))})
    print("BROWSER_ACCEPTANCE_PASSED",output,flush=True)

if __name__ == "__main__":
    main()
