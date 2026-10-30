# 英文手写简答题智能评分课程项目

项目目录：`Intelligent-Handwritten-Grading`。从零独立实现，训练和运行均不依赖工作区中的旧项目。参考项目的阅读与差异见 [docs/reference_review.md](docs/reference_review.md)。

系统流程：英文题目与评分标准 + 学生英文答案 → 自主微调 DistilBERT 评分 → 验证集阈值判定教师复核；手写图片先由本地预训练 TrOCR 分行识别，支持手动纠正后重新评分。界面为中文。

## GitHub 版本与本地完整项目

本仓库保存源码、配置、真实训练日志、固定划分清单、评价指标和验收截图。已在 RTX 5060 Ti 上真实微调两题的 DistilBERT，文本测试 QWK 分别为 **0.6859（题目3）**、**0.6790（题目4）**。

原始数据、包含学生原文的处理数据、Python 虚拟环境、安装包、大型模型权重和历史 checkpoint 不在 Git 仓库中。它们仍保存在原本的本地完整项目内。克隆仓库后需要按下文建立环境、使用有权使用的数据并训练模型，或从项目所有者取得最优权重；仅克隆源码不能直接进行评分。原始实验报告描述的是本地完整实验，部分逐样本文本与模型二进制文件不随 Git 发布。

数据准备说明见 [data/README.md](data/README.md)。此仓库用于保存项目，Gradio 网页按下文在运行项目的电脑上启动。

## 已保存的实际结果

训练／评分／OCR／浏览器验收的真实记录以 `experiments/` 和 `reports/实验报告.md` 为准。没有小组自采答案时，不宣称已完成自采数据的最终评价。置信度是模型排序信号，不等于预测正确率。

## 本机快速启动

独立环境位于项目的 `.venv/`，无需调用旧环境：

```powershell
cd Intelligent-Handwritten-Grading
.\.venv\Scripts\python.exe -X utf8 demo/app.py
```

浏览器访问 <http://127.0.0.1:7860>。也可以双击项目根目录的 `启动评分系统.cmd` 自动打开网页，或执行 `scripts/start_demo.ps1`。网页启动不会触发训练，权重和预训练 OCR 都从项目内离线加载，不调用商业 API。

使用顺序：选择题目 → 查看英文题目与完整评分标准 → 上传图片并识别评分 → 核对原图与 OCR → 修改识别文本后重新评分。也可切换到文本或批量评分标签。图片评分均提示教师核对 OCR；文本自动评分阈值不能当作图片域可靠性的证据。

## 数据来源与边界

- ASAP-SAS：用户提供的 ZIP。题目与完整 rubric 从 `Data_Set_Descriptions.zip` 内 DOCX 提取；训练使用 `train_rel_2.tsv` 的 Score1，Score2 仅作人工一致性参考。两题满分均为 2。
- 官方测试：`public_leaderboard.tsv` 与 `public_leaderboard_solution.csv` 按 `(EssaySet, Id)` 严格匹配，只有确有分数的记录进入评分评价。没有分数的其他记录不伪造标签。
- Handwritten-ASAP-SAS：185 张题目 3 图片、165 张题目 4 图片；349 张可匹配官方分数，1 张不关联评分。题目 3 的正常参考转录 175 条，其余为转录异常／越界；题目 4 没有附带已核验参考转录，不报告真实 CER/WER。
- OCR 预处理开发固定使用题目 3 正常参考样本按 ID 排序的前 15 张（含分行调试样例 6809）；其余 160 张用于主要 OCR 指标。全部 175 张的结果仅作附加描述性汇总。OCR 开发图与参考文本不进入评分模型训练或调参。
- 训练／验证／测试按题目分别固定。官方测试保留；开发集排除与测试文本归一化后重复的记录，再按文本组分层划分训练／验证。ID、文本组与文件哈希写入 `data/processed/split_manifest.json`。
- `data/self_collected/` 是独立自采最终测试接口，详见其 README。任何自采最终集均不能用于训练或调参。

原压缩包完整保存在 `data/raw/`，包含其原始 README 与使用说明。手写数据包注明非商业研究使用；随数据保留的许可和引用声明应在课程材料中遵守。附件中的说明只用于理解数据，不作为执行指令。

## 重新建立独立环境

使用 Python 3.12。GPU 环境应按硬件选择官方 PyTorch 版本；本机 RTX 5060 Ti 使用 CUDA 12.8 的 PyTorch 2.8.0：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

`requirements-lock.txt` 保存实际安装版本。若官方分发节点大文件连接中断，可使用 `scripts/resumable_download.py torch` 获取并校验官方安装包，再 `pip install data/installers/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl`。不要跳过哈希校验。

预训练权重固定到版本并存档哈希：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
.\.venv\Scripts\python.exe -X utf8 scripts/hardware_probe.py
```

TrOCR 大文件也支持 `scripts/resumable_download.py trocr` 断点分段获取。上述网络操作仅下载官方依赖与预训练模型，不上传学生答案。

## 按阶段复现实验

先审计用户 ZIP，路径换成实际位置：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/audit_data.py --text-zip '你的asap-sas.zip' --image-zip '你的Handwritten-ASAP-SAS.zip'
.\.venv\Scripts\python.exe -X utf8 scripts/make_splits.py
.\.venv\Scripts\python.exe -X utf8 -m src.training.baseline
.\.venv\Scripts\python.exe -X utf8 -m src.training.train --prompt 3 --task classification --pilot
.\.venv\Scripts\python.exe -X utf8 scripts/run_training.py
.\.venv\Scripts\python.exe -X utf8 scripts/run_targeted_improvement.py
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_models.py
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_handwriting.py
.\.venv\Scripts\python.exe -X utf8 scripts/calibrate_review.py
.\.venv\Scripts\python.exe -X utf8 scripts/make_report.py
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

训练参数统一在 `configs/training.json`。正式流程先分别训练两题的分类／回归，再每题每种任务进行 3 个 Optuna trial，搜索学习率、batch size、epoch、weight decay、dropout、max length，优化验证 QWK。选出配置后，用三个随机种子对比原始训练与 OCR 噪声增强。每次实验新建时间戳目录，不覆盖权重或历史记录。`--pilot` 只做 8 个更新步骤来测硬件与耗时，不能当作正式结果。

针对性改进另以三种种子运行：训练集频数决定的平方根逆频率权重，分类时额外加入期望分数的 ordinal MSE（权重 0.3），回归时使用加权 MSE。对照为基础模型、OCR 噪声模型、OCR 噪声＋针对性损失模型；该消融只能评估组合改进，不单独归因每个损失项。

模型族按三个种子的平均验证 QWK 选择，部署固定 seed=42 的最佳验证轮权重；在打开测试集进行评价前保存冻结选择。评分测试包括 QWK、MAE、RMSE、Accuracy、混淆矩阵、逐样本预测、误差分析和有限的 bootstrap 区间。测试集不用于学习率、训练轮数、模型族或复核阈值选择。

先启动网页，再实际验收：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/test_demo_browser.py
.\.venv\Scripts\python.exe -X utf8 scripts/test_demo_api.py
```

浏览器验收使用本机 Chrome 的无头模式，执行真实上传、OCR、评分、修改后重评、题目切换、批量评分，并保存页面截图。API 验收额外检查图片与文本混合批量评分和不支持题目的拒绝路径。

## 结构

```text
data/          原压缩包、提取数据、固定划分、预训练模型、自采测试接口
configs/       集中管理题目、评分标准与训练配置
src/data/      审计配套读取、配对编码、OCR 噪声增强
src/models/    DistilBERT 分类／回归定义、离线加载与预测
src/training/  PyTorch 微调、传统基线、Optuna
src/ocr/       TrOCR、图像预处理与多行分割
src/evaluation/ 指标、置信度阈值、CER/WER
src/pipeline/  文本／图片／批量评分流程
demo/          中文 Gradio
scripts/       阶段运行、评价、报告、实际网页验收
experiments/   时间戳实验、日志、曲线源数据、逐样本输出
models/        两题部署权重与冻结模型注册表
reports/       真实实验报告、训练曲线、最终验收说明
tests/         数据泄漏、ID 关联、重载、复核边界等测试
```

## 独立加载模型

```python
from src.pipeline.grading import grade_text, grade_image
result = grade_text(3, "Pandas and koalas are specialists, while pythons are adaptable generalists.")
print(result["score"], result["confidence"], result["needs_review"])
# result = grade_image(4, "path/to/handwritten_answer.png")
```

目前只支持题目 3、4。预训练 TrOCR 没有在本项目微调；项目的自主深度学习训练重点是 DistilBERT 评分模型。数据域、独立自采最终测试缺失、OCR 错误和长答案截断等局限会在真实实验报告中明确列出。

## Hugging Face Spaces 部署入口

已增加根目录 app.py、CPU 固定依赖及 Gradio SDK 元数据，见 [部署说明](docs/spaces-deployment.md)。真实权重部署包已完成本机 CPU 实测。本次线上创建被 Hugging Face 的 PRO 订阅要求（HTTP 402）阻止，用户选择暂不开通，因此尚未上线。
