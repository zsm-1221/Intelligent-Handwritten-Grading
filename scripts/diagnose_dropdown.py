import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright,expect
from src.common import experiment,write_json
out=experiment("dropdown_diagnostic")
with sync_playwright() as p:
    browser=p.chromium.launch(channel="chrome",headless=True)
    page=browser.new_page(viewport={"width":1440,"height":1100},locale="zh-CN")
    page.goto("http://127.0.0.1:7860",wait_until="domcontentloaded")
    expect(page.get_by_label("选择题目",exact=True)).to_be_visible(timeout=60000)
    field=page.get_by_label("选择题目",exact=True)
    field.click()
    field.fill("题目 4")
    field.press("ArrowDown")
    page.wait_for_timeout(1000)
    page.screenshot(path=str(out / "dropdown.png"),full_page=True)
    (out / "page.html").write_text(page.content(),encoding="utf-8")
    print("OPTIONS",page.get_by_role("option").all_text_contents(),flush=True)
    print(page.locator("body").inner_text()[:4000],flush=True)
    print("DIAGNOSTIC",out,flush=True)
    browser.close()
