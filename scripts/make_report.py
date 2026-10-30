import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common import ROOT, read_json, write_json

def main():
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    ep = ROOT / read_json(ROOT / "experiments/evaluation_pointer.json")["path"]
    hp = ROOT / read_json(ROOT / "experiments/handwriting_pointer.json")["path"]
    rp = ROOT / read_json(ROOT / "experiments/review_pointer.json")["path"]
    evaluation,handwriting = read_json(rp / "test_routing.json"),read_json(hp / "metrics.json")
    registry = read_json(ROOT / "models/registry.json")
    stability = read_json(ep / "seed_stability.json")
    manifest = read_json(ROOT / "data/processed/split_manifest.json")
    comparison = read_json(ep / "comparison.json")
    lines = ["# 实验与验收报告", "", "全部数值由本地真实实验记录生成。部署选择在测试集评价之前冻结；测试指标不用于继续调参。", "", "## 数据与划分", "", "两道题均为 0～2 分，以 Score1 为目标。train_rel_2.tsv 作为开发数据，公开测试文本按题号和 ID 与官方 solution 严格关联。去除开发数据中与测试集规范化文本重复的记录，训练／验证按文本组分层划分。", "", "| 题目 | 训练 | 验证 | 官方测试 | 排除的测试重复开发记录 |", "|---|---:|---:|---:|---:|"]
    for p in ("3","4"):
        m=manifest[p]
        lines.append(f"| {p} | {m['counts']['train']} | {m['counts']['validation']} | {m['counts']['test']} | {m['excluded_test_duplicate_rows']} |")
    lines += ["", "350 张手写图中，349 张有严格 ID 匹配的人工分数；未匹配的图不赋予分数。Prompt 3 有 175 张正常参考转录，其中按 ID 固定前 15 张供通用分行预处理开发，余下 160 张用于主 CER/WER；另外 10 张被标为转录异常或越界，不纳入主 OCR 指标。OCR 开发集没有参与评分模型训练或调参。Prompt 4 未附参考转录，不报告其真实 CER/WER。", "", "## 评分模型", "", "| 题目 | 部署任务 | 测试 QWK | MAE | RMSE | Accuracy | QWK bootstrap 95% CI |", "|---|---|---:|---:|---:|---:|---|"]
    for p in ("3","4"):
        r=evaluation[p]; m=r["test"]
        lines.append(f"| {p} | {registry[p]['task']} | {m['qwk']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['accuracy']:.4f} | {r['test_qwk_bootstrap_95ci']} |")
    lines += ["", "bootstrap 区间对当前冻结预测按样本重采样，不能反映所有训练随机性或跨学校泛化。每题先正式微调分类／回归，再各做 3 次 Optuna 受控搜索（共 12 个 trial），搜索学习率、batch、epoch、weight decay、dropout 和 max length。搜索预算小，不宣称全局最优。", "", "原问题与版本化紧凑评分标准占输入前半预算，后半留给答案。完整 rubric 在 configs/prompts.json 和网页中保留。所有 DistilBERT 层参与训练，BF16（支持时）、AdamW、线性 warmup/scheduler、gradient clipping、early stopping，按验证 QWK 保存最佳轮。", "", "### 基线与分类／回归初始对比", "", "| 题目 | 方法 | 验证 QWK | 测试 QWK | 测试 MAE |", "|---|---|---:|---:|---:|"]
    for r in comparison:
        if r["stage"] in ("baseline","formal_initial"):
            lines.append(f"| {r['prompt']} | {r['method']} | {r['validation_qwk']:.4f} | {r['qwk']:.4f} | {r['mae']:.4f} |")
    lines += ["", "### 随机种子与 OCR 噪声消融", "", "在各题验证集选出的任务和配置上，分别以 42、123、2026 三个种子训练原始文本与通用 OCR 字符混淆／丢字符增强模型。每次取样以 50% 概率使用噪声副本，保持训练样本数及每轮优化步数相同；两组都使用相同早停规则，实际训练轮数可能不同。增强仅使用训练集，噪声规则不来自测试图。按三个种子平均验证 QWK 选择模型族，部署该族预先指定 seed=42 的模型。", "", "| 题目 | 模型族 | 验证 QWK 均值 ± 样本标准差 | 测试 QWK 均值 ± 样本标准差 |", "|---|---|---|---|"]
    for p in ("3","4"):
        for name,f in stability[p]["families"].items():
            values=[r["qwk"] for r in comparison if str(r["prompt"])==p and r["stage"]==name]
            lines.append(f"| {p} | {name} | {f['mean_validation_qwk']:.4f} ± {f['std_validation_qwk']:.4f} | {np.mean(values):.4f} ± {np.std(values,ddof=1):.4f} |")
    lines.append("")
    for p in ("3","4"):
        lines.append(f"题目 {p} 最终选择 `{stability[p]['chosen_family']}`。这是当前预算下平均验证表现的选择，不证明某种增强方法普遍有效或无效；三种种子的差异不等同于统计显著性检验。")
    lines += ["", "repeat_clean 为原始训练，repeat_noise 为同预算噪声增强，repeat_targeted 为噪声增强加针对性损失：训练集频数计算平方根逆频率权重；分类另加 0.3 倍的期望分数 MSE，回归采用加权 MSE。该项只能归因于组合改进，未单独分离各损失的因果贡献。增强是否有效以表中实际结果为准，不强制宣称改进。各次训练仅保证随机种子、版本与记录可追溯，GPU 运算不保证跨设备逐位一致。", "", "## OCR 与端到端评价", ""]
    for p in ("3","4"):
        h=handwriting[p]; v=h["verified_ocr"]
        lines.append(f"- 题目 {p}：识别 {h['images']} 张；具有正常参考转录的样本 {v['n']}，CER={v['cer']}，WER={v['wer']}。")
        for label,pair in [("核验参考转录配对",h["verified_reference_pair"]),("原始 ASAP 文本代理配对",h["official_original_text_pair"])]:
            a=pair.get("reference_grading",pair.get("original_grading")); b=pair["ocr_grading"]
            lines.append(f"- 题目 {p} {label}：n={a['n']}，文本 QWK/MAE={a['qwk']}/{a['mae']}，OCR QWK/MAE={b['qwk']}/{b['mae']}。")
            effect=pair.get("paired_effect",{})
            if effect.get("n",0):
                lines.append(f"- 该配对中，OCR 导致预测分数变化的比例为 {effect['score_change_rate']:.2%}；绝对评分误差增加 {effect['worsened_absolute_error_rate']:.2%}，偶然降低 {effect['improved_absolute_error_rate']:.2%}，平均新增绝对误差 {effect['mean_added_absolute_error']:.4f}。")
                if "spearman_cer_vs_added_absolute_error" in effect:
                    lines.append(f"- CER 与新增绝对评分误差的 Spearman 相关为 {effect['spearman_cer_vs_added_absolute_error']}；按 CER 分组的真实评分指标见 OCR metrics.json，不把所有字符错误视为同等影响。")
    lines += ["", "CER/WER 采用 NFKC、小写、空白归一化，保留标点，以参考字符／词总数做微平均；OCR 使用官方 TrOCR-small-handwritten 权重，按水平投影分行后批量识别。未在本项目微调 OCR。Prompt 4 的原始 ASAP 文本仅是代理对照，不能据此证明转录一致。手写数据是对原答案的转写，不等同于真实课堂独立作答，域差异仍需自采测试。", "", "## 评分置信度与复核", "", "模型 softmax 最大值／回归输出距最近合法整数的启发式量只用于排序，不是评分正确率。阈值只由验证集选择，要求至少 30 个自动接受样本、精确一致率 ≥85%、误差不超过 1 分的一致率 ≥98%；找不到满足条件的阈值则全部复核。该经验规则不是可靠性保证，阈值选择与模型选择共用验证集，存在乐观偏差。", ""]
    for p in ("3","4"):
        r=evaluation[p]["routing"]; policy=registry[p]["review"]
        lines.append(f"- 题目 {p}：阈值 {policy['threshold']:.6f}；测试自动覆盖率 {r['automatic_coverage']:.2%}，复核覆盖率 {r['review_coverage']:.2%}；自动子集实际精确一致率 {r['automatic']['accuracy']}，误差≤1分一致率 {r['automatic']['within_one']}。")
        if r["automatic"]["accuracy"] is not None and r["automatic"]["accuracy"] < policy["target_exact"]:
            lines.append(f"- 题目 {p} 的测试自动子集精确一致率低于验证阶段的 {policy['target_exact']:.0%} 目标；保留该结果，不利用测试集反调阈值。阈值不是正确率保证。")
    lines += ["", "上述覆盖率针对文本评分；手写 OCR 的错误会改变分数，因此 Demo 对图片评分均提示教师核对。空白、非英文、过长、OCR 异常及不支持题目有明确处理。", "", "## 产物位置", "", f"- 评分评价：`{ep.relative_to(ROOT).as_posix()}`，包含逐样本测试预测、混淆矩阵、误差 CSV、全部对比和随机种子结果。", f"- OCR 评价：`{hp.relative_to(ROOT).as_posix()}`，包含每张图识别文本、行框、参考状态、CER/WER 计数和评分变化。", "- 权重与冻结版本：`models/registry.json` 及其指向的独立题目目录。", "- 训练记录：`experiments/*/run.json`、`history.json`、`summary.json`、`checkpoint/`。", "- 自采最终测试：`data/self_collected/`，当前没有真实小组自采样本；接口可用，但不编造该项完成或指标。", "", "## 局限与下一步", "", "这是受控的小样本课程实验。公开测试集可用于复现实验，不是盲测；没有独立学校／学生域的泛化证明。三种随机种子的稳定性证据有限。完整 rubric 压缩、长答案截断、分行不准和 OCR 领域差异均会影响分数。下一步应在冻结模型后收集独立自采数据，由教师标注并评价，不能据此再次调参后仍称独立测试。", ""]
    (out / "实验报告.md").write_text("\n".join(lines),encoding="utf-8")
    all_summaries=[read_json(p) for p in (ROOT / "experiments").glob("*/summary.json")]
    cost={"hardware":read_json(ROOT / "experiments/hardware.json"),"training_run_count_including_pilot_and_pruned":len(all_summaries),
          "optimizer_updates":sum(r["optimizer_steps"] for r in all_summaries),
          "sum_training_run_elapsed_seconds":sum(r["elapsed_seconds"] for r in all_summaries),
          "note":"Measured run elapsed includes epoch validation and checkpoint writes, excludes dependency download/setup and OCR/demo time; no monetary cost is inferred."}
    write_json(out / "compute_cost.json",cost)
    with (out / "实验报告.md").open("a",encoding="utf-8") as f:
        f.write(f"\n## 实测算力记录\n\n本项目累计记录 {cost['training_run_count_including_pilot_and_pruned']} 次训练（含短程试验和剪枝），实际执行 {cost['optimizer_updates']} 次优化器更新；各训练运行记录耗时合计 {cost['sum_training_run_elapsed_seconds']/60:.2f} 分钟，包含轮次验证和权重保存，不含安装下载、OCR 和网页验收。硬件为 {cost['hardware'].get('gpu','CPU')}，详细记录见 compute_cost.json。\n\n![训练曲线](training_curves.png)\n\n![真实模型比较](model_comparison.png)\n")
    # Real training curves for deployed models.
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for j,p in enumerate(("3","4")):
        h=read_json(ROOT / "experiments" / registry[p]["version"] / "history.json")
        epochs=[r["epoch"] for r in h]
        axes[0,j].plot(epochs,[r["train_loss"] for r in h],"o-",label="Train")
        axes[0,j].plot(epochs,[r["validation"]["loss"] for r in h],"s-",label="Validation")
        axes[0,j].set(title=f"Prompt {p}: Loss",xlabel="Epoch",ylabel="Loss")
        axes[1,j].plot(epochs,[r["validation"]["qwk"] for r in h],"o-",label="Validation QWK")
        axes[1,j].set(title=f"Prompt {p}: Validation QWK",xlabel="Epoch",ylabel="QWK")
        axes[0,j].legend();axes[1,j].legend()
    fig.tight_layout();fig.savefig(out / "training_curves.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,p in zip(axes,("3","4")):
        methods=["tfidf_logistic","tfidf_ridge","repeat_clean","repeat_noise","repeat_targeted"]
        means,errors=[],[]
        for method in methods:
            values=[r["qwk"] for r in comparison if str(r["prompt"])==p and (r["method"]==method or r["stage"]==method)]
            means.append(float(np.mean(values)));errors.append(float(np.std(values,ddof=1)) if len(values)>1 else 0.)
        ax.bar(range(5),means,yerr=errors,capsize=4,color=["#8196a8","#a6b7c5","#277f8e","#49a7a4","#c49448"])
        ax.set(xticks=range(5),xticklabels=["TF-IDF LR","TF-IDF Ridge","Clean","OCR noise","Targeted"],ylabel="Test QWK",title=f"Prompt {p}")
        ax.tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(out / "model_comparison.png",dpi=180);plt.close(fig)
    if (ROOT / "experiments/browser_pointer.json").exists() and (ROOT / "experiments/api_pointer.json").exists():
        import shutil
        import xml.etree.ElementTree as ET
        bp=ROOT / read_json(ROOT / "experiments/browser_pointer.json")["path"]
        ap=ROOT / read_json(ROOT / "experiments/api_pointer.json")["path"]
        browser=read_json(bp / "results.json");api=read_json(ap / "results.json")
        assert browser["passed"] and api["passed"]
        suites=list(ET.parse(out / "pytest.xml").getroot().iter("testsuite"))
        count=sum(int(s.attrib.get("tests",0)) for s in suites)
        failed=sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0)) for s in suites)
        assert failed==0
        for source,name in [("02_image_graded.png","demo_image.png"),("04_batch.png","demo_batch.png"),("05_prompt4.png","demo_prompt4.png")]:
            shutil.copy2(bp / source,out / name)
        qa=["# 最终运行验收","",f"功能测试：{count} 项通过，{failed} 项失败；包含最佳权重重载与验证集预测复现。","","Chrome 真实浏览器：",""]
        qa += ["- "+step["test"]+"：通过" for step in browser["steps"]]
        qa += ["","本地 HTTP：",""]+["- "+step["test"]+"：通过" for step in api["results"]]
        qa += ["", "自采接口：尚无真实小组采集样本，空输入保护已实测，不生成虚构指标。", "", "开机后双击项目根目录的 `启动评分系统.cmd`，或按 README 启动；无需重新训练。", "", "![真实手写评分](demo_image.png)", "", "![真实批量评分](demo_batch.png)", "", "![题目4评分](demo_prompt4.png)"]
        (out / "验收记录.md").write_text("\n".join(qa),encoding="utf-8")
        with (out / "实验报告.md").open("a",encoding="utf-8") as f:
            f.write(f"\n## 实际运行验收\n\n{count} 项功能测试通过；Chrome 完成真实图片上传、OCR、评分、手动修改重评、批量评分与题目切换，HTTP 完成图片／文本混合批量与不支持题目拒绝测试。详见 [验收记录](验收记录.md)。\n\n其他候选即使测试 QWK 更高，也不改变按验证集预先冻结的部署规则。\n")
    print(out / "实验报告.md")

if __name__ == "__main__":
    main()
