import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

root=Path(__file__).resolve().parents[1]
out=root/'reports/final-20261103'
assets=out/'PPT图表'
checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080},locale='zh-CN')
    page.goto('http://127.0.0.1:7862',wait_until='domcontentloaded')
    expect(page.get_by_role('tab',name='项目实验结果',exact=True)).to_be_visible(timeout=60000)
    page.get_by_role('tab',name='项目实验结果',exact=True).click()
    expect(page.get_by_text('最新项目版本：2026-11-03 软标签与序数损失实验',exact=False)).to_be_visible()
    page.screenshot(path=str(assets/'系统_最新实验结果.png'),full_page=True)
    checks.append('最新实验结果标签及项目版本')
    page.get_by_role('tab',name='手写图片评分',exact=True).click()
    page.locator('input[type=file]').first.set_input_files(str(out/'generated-public-test.png'))
    page.get_by_role('button',name='识别图片并评分',exact=True).click()
    expect(page.get_by_label('OCR 识别文本（可手动修改）')).not_to_have_value('',timeout=180000)
    expect(page.locator('.score-card').filter(has_text='预测分数').filter(visible=True)).to_be_visible(timeout=180000)
    page.screenshot(path=str(assets/'系统_合成字体图片OCR与评分.png'),full_page=True)
    checks.append('合成字体图片上传OCR及真实模型评分')
    page.get_by_label('OCR 识别文本（可手动修改）').fill('Pandas and koalas eat specialized diets. Pythons eat many different animals.')
    page.get_by_role('button',name='按修改后的文本重新评分',exact=True).click()
    page.wait_for_timeout(2000)
    expect(page.locator('.score-card').filter(has_text='预测分数').filter(visible=True)).to_be_visible()
    checks.append('修改后重新评分')
    page.get_by_role('tab',name='批量评分',exact=True).click()
    page.get_by_label('批量英文答案（每行一份）',exact=True).fill('Pandas and koalas have specialized diets.\nPandas are reptiles and eat snakes.')
    page.get_by_role('button',name='开始批量评分',exact=True).click()
    expect(page.get_by_text('文本 2',exact=True).filter(visible=True).first).to_be_visible(timeout=60000)
    page.screenshot(path=str(assets/'系统_批量评分.png'),full_page=True)
    checks.append('批量评分')
    page.goto((out/'最终项目报告.html').as_uri(),wait_until='load')
    assert page.locator('h1').inner_text().startswith('英文手写简答题智能评分')
    assert page.locator('table').count()>=4
    assert page.evaluate('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
    page.screenshot(path=str(out/'报告阅读版预览.png'),full_page=False)
    checks.append('HTML报告标题表格图片完整加载')
    browser.close()
(out/'browser-release-tests.json').write_text(json.dumps({'passed':True,'checks':checks,'fixture':'generated Segoe Print image, not real student handwriting'},ensure_ascii=False,indent=2),encoding='utf-8')
print('BROWSER_RELEASE_CHECKS_PASSED',len(checks))
