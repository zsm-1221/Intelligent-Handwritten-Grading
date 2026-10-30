"""Chinese local Gradio UI. Training is never triggered by UI requests."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
import html
import gradio as gr
from src.common import ROOT, prompt_info
from src.pipeline.grading import grade_text, grade_image, grade_batch

CSS = """
.gradio-container {max-width:1180px!important; font-family:'Microsoft YaHei',sans-serif!important;}
.hero {background:linear-gradient(120deg,#133e65,#126e82);color:white;padding:28px 32px;border-radius:18px;margin-bottom:16px;}
.hero h1 {color:white!important;font-size:28px!important;margin:0 0 10px!important;}
.hero p {color:#dfedf2!important;margin:0!important;}
.score-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0;}
.score-card {background:#f1f7fa;border:1px solid #d7e6ed;border-radius:12px;padding:18px;}
.score-card strong {display:block;font-size:26px;color:#124b67;margin-top:8px;}
.review {padding:14px 18px;border-radius:10px;background:#fff3d8;color:#755010;}
.okay {padding:14px 18px;border-radius:10px;background:#e5f5ed;color:#1d6245;}
footer {display:none!important;}
"""
JS = """() => {
 document.documentElement.lang='zh-CN';
 const words={'Drop Image Here':'将图片拖放到此处','Click to Upload':'点击上传','- or -':'或','Drop Files Here':'将文件拖放到此处','Clear':'清空','Download':'下载','Fullscreen':'全屏','Share':'分享','Processing...':'正在处理…','Error':'错误','Submit':'提交','Upload':'上传','View API':'接口说明','Use via API':'接口说明','Settings':'设置','Built with Gradio':'中文手写简答题评分系统'};
 const translate=()=>{const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let n;while(n=walker.nextNode()){const t=n.nodeValue.trim();if(words[t])n.nodeValue=n.nodeValue.replace(t,words[t]);} document.querySelectorAll('[title],[aria-label]').forEach(e=>{for(const a of ['title','aria-label'])if(words[e.getAttribute(a)])e.setAttribute(a,words[e.getAttribute(a)]);});};
 new MutationObserver(translate).observe(document.body,{childList:true,subtree:true});translate();
}"""

def show_prompt(prompt):
    p = prompt_info(prompt)
    return f"### 英文题目（满分 {p['max_score']} 分）\n\n{p['question']}", "### 英文评分标准\n\n" + p["rubric"]

def render_result(result):
    s = result
    score = f"<div class='score-grid'><div class='score-card'>预测分数<strong>{s['score']} / {s['max_score']}</strong></div><div class='score-card'>评分等级<strong>{s['level']}</strong></div><div class='score-card'>评分置信度信号<strong>{s['confidence']:.3f}</strong></div></div>"
    label = "建议教师复核" if s["needs_review"] else "达到文本自动评分阈值"
    detail = "；".join(s["review_reasons"]) if s["needs_review"] else "验证集阈值下达到预设一致性要求，仍需结合教学情境使用。"
    score += f"<div class='{'review' if s['needs_review'] else 'okay'}'><b>{label}</b><br>{html.escape(detail)}</div>"
    note = f"模型版本：{s['version']}　｜　本次推理耗时：{s['seconds']:.2f} 秒\n\n评分置信度是模型排序信号，**不是预测正确率**。"
    if s["review_threshold"] > 1:
        note += "验证集中未找到达到预设一致性要求的自动评分阈值，因此全部建议复核。"
    else:
        note += f"当前复核阈值为 {s['review_threshold']:.4f}，仅由验证集选择。"
    if s.get("ocr"):
        note += f"\n\n手写识别：{len(s['ocr']['lines'])} 行，识别耗时 {s['ocr']['seconds']:.2f} 秒。识别错误可能改变分数，请核对原图。"
    return score, note

def ui_image(prompt,image):
    try:
        result = grade_image(prompt,image)
        return result["answer"], *render_result(result), result["ocr"]
    except Exception as exc:
        raise gr.Error(str(exc))

def ui_text(prompt,text,ocr_state=None):
    try:
        result = grade_text(prompt,text,ocr_state)
        return render_result(result)
    except Exception as exc:
        raise gr.Error(str(exc))

def ui_batch(prompt,text,files):
    try:
        return grade_batch(prompt,text,files)
    except Exception as exc:
        raise gr.Error(str(exc))

def build_app(public_demo=False):
    theme=gr.themes.Soft(primary_hue="teal",font=["Microsoft YaHei","Arial","sans-serif"],font_mono=["Consolas","monospace"])
    with gr.Blocks(title="英文手写简答题智能评分",theme=theme,css=CSS,js=JS,
                   analytics_enabled=False,delete_cache=(3600,3600) if public_demo else None) as app:
        gr.HTML("<div class='hero'><h1>英文手写简答题智能评分</h1><p>手写识别 · 自主训练评分 · 教师复核　｜　深度学习课程项目</p></div>")
        gr.Markdown("上传英文手写答案，或直接输入英文文本。系统支持题目 3 和题目 4，满分均为 2 分。请先核对题目，再进行评分。")
        if public_demo:
            gr.Markdown("公开课程演示，请使用不含姓名、学号等个人信息的匿名答案。上传内容不参与模型训练；临时文件定期清理。首次加载模型和手写识别可能需要等待，请勿重复提交。")
        prompt = gr.Dropdown(choices=[("题目 3：熊猫、考拉与蟒蛇",3),("题目 4：入侵物种",4)], value=3,label="选择题目",interactive=True,
                             allow_custom_value=True,info="请选择题目 3 或题目 4；其他题目需要教师人工复核。")
        q,r = show_prompt(3)
        question = gr.Markdown(q)
        with gr.Accordion("查看完整英文评分标准",open=False):
            rubric = gr.Markdown(r)
        prompt.change(show_prompt,prompt,[question,rubric],api_name="select_prompt")
        with gr.Tab("手写图片评分"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil",label="上传英文手写答案图片",sources=["upload"],height=330)
                    image_button = gr.Button("识别图片并评分",variant="primary")
                with gr.Column():
                    transcription = gr.Textbox(label="OCR 识别文本（可手动修改）",lines=9,placeholder="识别结果显示在这里；核对并修改后可以重新评分。",interactive=True)
                    edit_button = gr.Button("按修改后的文本重新评分")
            ocr_state = gr.State(None)
            image_result = gr.HTML()
            image_note = gr.Markdown()
            image_button.click(ui_image,[prompt,image],[transcription,image_result,image_note,ocr_state],api_name="grade_image",concurrency_limit=1)
            edit_button.click(ui_text,[prompt,transcription,ocr_state],[image_result,image_note],api_name="regrade_text",concurrency_limit=1)
        with gr.Tab("直接输入文本评分"):
            text = gr.Textbox(label="英文学生答案",lines=6,placeholder="请输入与当前题目对应的英文答案。")
            text_button = gr.Button("开始文本评分",variant="primary")
            text_result,text_note = gr.HTML(),gr.Markdown()
            text_button.click(lambda p,t:ui_text(p,t),[prompt,text],[text_result,text_note],api_name="grade_text",concurrency_limit=1)
        with gr.Tab("批量评分"):
            gr.Markdown("每行输入一份英文答案，或上传多张手写答案图片。本批答案均使用上方选择的题目。")
            batch_text = gr.Textbox(label="批量英文答案（每行一份）",lines=5)
            batch_files = gr.File(label="批量上传手写图片",file_count="multiple",file_types=["image"],type="filepath")
            batch_button = gr.Button("开始批量评分",variant="primary")
            table = gr.Dataframe(label="批量评分结果",interactive=False,wrap=True)
            batch_button.click(ui_batch,[prompt,batch_text,batch_files],table,api_name="grade_batch",concurrency_limit=1)
        gr.Markdown("课程演示用途。评分模型由本项目训练，图片识别使用本地 TrOCR。图片识别有误时，请修改文本后重新评分；独立自采答案的最终成绩请由教师核验。")
    return app

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--open-browser",action="store_true")
    args=parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1",server_port=7860,share=False,inbrowser=args.open_browser,show_api=False)
