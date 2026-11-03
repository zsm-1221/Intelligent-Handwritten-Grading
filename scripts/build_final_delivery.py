"""Build an evidence-bound course report and a curated slide asset directory."""
import json, shutil, hashlib, html
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/soft-ordinal-20261103'
OUT=ROOT/'reports/final-20261103'
OUT.mkdir(exist_ok=True)
ASSETS=OUT/'PPT图表';ASSETS.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
freeze=read(EXP/'selection_frozen.json');assert read(EXP/'complete.json')['finished']
df=pd.read_csv(EXP/'comparison.csv')
names={'hard_ce':'硬标签 CE','soft_ce':'软标签 CE','hard_ce_emd':'硬标签 CE + EMD','soft_ce_emd':'软标签 CE + EMD'}
rows=[]
for (p,g),v in df.groupby(['prompt','group'],sort=False):
    rows.append({'题目':int(p),'方法':names[g],'验证QWK':f"{v.validation_qwk.mean():.4f} ± {v.validation_qwk.std():.4f}",
                 '测试QWK':f"{v.test_qwk.mean():.4f} ± {v.test_qwk.std():.4f}",'测试MAE':f'{v.test_mae.mean():.4f}',
                 '测试RMSE':f'{v.test_rmse.mean():.4f}','测试准确率':f'{v.test_accuracy.mean():.2%}'})
summary=pd.DataFrame(rows);summary.to_csv(OUT/'四组消融汇总.csv',index=False,encoding='utf-8-sig')

def mdtable(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in frame.itertuples(index=False,name=None))

errors=[];details=[]
for p in [3,4]:
    group=freeze['champions'][str(p)]
    for seed in [42,123,2026]:
        t=read(EXP/f'repeats/p{p}_{group}_s{seed}/test.json')
        assert t['selection_sha256']==read(EXP/'complete.json')['test_selection_sha256']
        # Exported public artifacts contain contingency counts, not student rows.
        cm=np.array(t['metrics']['confusion_matrix'])
        pairs=[(a,b) for a in range(3) for b in range(3) for _ in range(int(cm[a,b]))]
        a,b=np.array(pairs).T;delta=b-a
        details.append(dict(prompt=p,group=group,seed=seed,n=len(a),over_score_rate=float((delta>0).mean()),under_score_rate=float((delta<0).mean()),two_point_error_rate=float((abs(delta)==2).mean()),signed_bias=float(delta.mean())))
        for score in [0,1,2]:
            mask=a==score
            errors.append(dict(prompt=p,group=group,seed=seed,true_score=score,n=int(mask.sum()),recall=float((b[mask]==score).mean()),mae=float(abs(delta[mask]).mean())))
pd.DataFrame(errors).to_csv(OUT/'按真实分数误差明细.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(details).to_csv(OUT/'高估低估与跨级误差.csv',index=False,encoding='utf-8-sig')
et=pd.DataFrame(errors).groupby(['prompt','group','true_score']).agg(n=('n','first'),recall=('recall','mean'),mae=('mae','mean')).reset_index().round(4)
dt=pd.DataFrame(details).groupby('prompt').mean(numeric_only=True).drop(columns=['seed']).round(4)
configrows=[]
for key,r in freeze['selected'].items():
    c=r['config'];configrows.append(dict(prompt=r['prompt'],group=r['group'],lr=c['learning_rate'],batch=c['batch_size'],epochs=c['epochs'],dropout=c['dropout'],max_length=c['max_length'],weight_decay=c['weight_decay']))
pd.DataFrame(configrows).to_csv(OUT/'验证集冻结最优配置.csv',index=False,encoding='utf-8-sig')

manifest=[]
def asset(source,stem,slide,caption):
    copied=[]
    for ext in ['png','pdf','svg']:
        path=source.with_suffix('.'+ext)
        if path.exists():
            target=ASSETS/(stem+'.'+ext);shutil.copy2(path,target);copied.append(target.name)
    manifest.append({'编号':stem,'建议页码':slide,'说明':caption,'文件':'；'.join(copied)})
for p in [3,4]:
    for metric,zh in [('validation_qwk','验证集选模'),('test_qwk','测试QWK与种子稳定性'),('test_mae','测试MAE'),('test_accuracy','测试准确率')]:
        asset(EXP/f'figures/p{p}_{metric}',f'P{p}_{metric}',9 if metric=='validation_qwk' else 10,f'题目{p}：{zh}。柱为3种子均值，误差棒为样本标准差，黑点为每个种子；不是置信区间。')
    asset(EXP/f'figures/p{p}_search_budget',f'P{p}_搜索预算',8,'每组同样20组候选；横轴是共享随机候选顺序，纵轴为迄今最佳验证QWK。')
    group=freeze['champions'][str(p)]
    for suffix,zh in [('loss','训练验证损失'),('validation_qwk','逐轮验证QWK'),('confusion','测试混淆矩阵')]:
        asset(EXP/f'figures/p{p}_{group}_s42_{suffix}',f'P{p}_选定方法_seed42_{suffix}',11 if suffix=='confusion' else 7,f'题目{p}，{names[group]}，预先固定seed42的示例：{zh}。单种子不代表三种子平均。')
asset(ROOT/'reports/model_comparison','早期分类回归与传统基线',6,'早期独立实验阶段的传统基线与分类/回归模型比较；不能与新消融混为同一搜索预算。')
pd.DataFrame(manifest).to_csv(ASSETS/'图表索引.csv',index=False,encoding='utf-8-sig')
(ASSETS/'使用说明.md').write_text('# PPT图表索引\n\nPNG适合直接插入PPT；SVG/PDF适合矢量编辑。每张图只选一种格式。所有数值来自实际实验。\n\n'+mdtable(pd.DataFrame(manifest)),encoding='utf-8')

body=f'''# 英文手写简答题智能评分：从双评分者学习到教师复核

## 摘要

本项目研究如何将英文手写简答题识别为文本、依据题目和评分标准自动评分，并提示评分可靠性不足的答案交由教师复核。系统以PyTorch微调DistilBERT，用预训练TrOCR进行多行手写识别，提供全中文Gradio界面。仅研究有手写图像支持的ASAP-SAS题目3、4，分数范围均为0—2。

在保留原划分及部署权重的条件下，新增双评分者软标签与序数EMD损失的2×2消融。每题每组共享20组候选配置，完成160次搜索训练、24次三种子重复，并在冻结验证集选模后进行测试。题目4软标签CE的测试QWK均值为0.6722，硬标签CE为0.6018；题目3各方法差异较小。EMD没有稳定带来额外收益。已有部署模型的题目3/4文本测试QWK为0.6859/0.6790；不因新增实验某个测试值较高而自动替换。OCR仍是端到端性能瓶颈，本系统定位为有教师复核的课程原型。

## 1. 明确问题与研究目标

RQ1：在同一文本划分上，Transformer是否优于TF-IDF传统基线？RQ2：两位人工评分者存在分歧时，用软标签代替Score1硬标签能否提高评分表现及稳定性？RQ3：显式考虑0、1、2分有序关系的EMD损失是否有额外收益？RQ4：手写识别错误如何传递到评分，以及哪些输出应由教师复核？

课程重点是自主完成深度学习评分器训练、优化与评价。TrOCR使用预训练权重，未在本项目重新训练；没有商业大模型API、随机分数或固定分数替代评分模型。

## 2. 数据审计、固定划分与泄漏控制

|题目|训练|验证|测试|满分|
|---|---:|---:|---:|---:|
|3|1447|361|406|2|
|4|1325|332|295|2|

开发文本来自ASAP-SAS，题目与完整rubric从数据说明提取。官方测试文本和分数按题号与ID严格关联，训练/验证按归一化文本组分层划分，排除与测试重复的开发文本。新增实验直接读取原划分，不重新随机切分；完成标记确认原划分文件、模型registry和部署权重的SHA256未改变。

手写图像共350张（题目3为185张、题目4为165张），349张严格匹配评分，1张无法匹配的图片不强行赋分。题目3有175条正常参考转录，其中15张固定用于OCR预处理开发，其余160张用于主要OCR评价；题目4没有已核验参考转录，因此不报告CER/WER。自采目录与标注接口保留，目前没有真实小组自采最终样本。

训练集两位评分者分歧：题目3为284/1447（19.63%），题目4为237/1325（17.89%）；验证集分别为86/361与61/332。分歧体现人工标注不确定性，不代表某位评分者必然错误。

## 3. 项目步骤与系统架构

数据审计与固定划分 → TF-IDF基线 → DistilBERT分类/回归微调 → 早期小预算搜索、噪声增强与针对性改进 → 验证集冻结选模 → 文本及OCR配对评价 → 复核阈值 → 中文Demo → 新一轮软标签/序数损失公平搜索与三种子消融 → 冻结选择、测试和误差分析。

推理流程为：选择题目 → 输入英文答案或上传手写图 → 图像预处理及分行TrOCR → 题目+紧凑评分标准+学生答案进入DistilBERT → 合法分数 → 可靠性信号与教师复核提示。教师可以修改OCR结果后重评，或批量评分。训练代码与Demo分离，网页请求不触发训练。

## 4. 深度学习模型与优化方法

DistilBERT从项目内固定预训练版本初始化，所有层参与微调，三分类头对应0、1、2分；早期阶段另训练单输出回归头。输入前半长度预算给英文题目与紧凑rubric，后半留给答案，使用相同编码规则验证和推理。紧凑标准与完整rubric不是同一文本，压缩信息和截断是局限。

设两位评分者给出的分数为s₁、s₂，K=3。软目标为 yₖ=0.5·1[s₁=k]+0.5·1[s₂=k]；一致时等于独热标签，分歧时在两类之间分配概率质量。缺失Score2回退Score1，非法有限分数拒绝处理。评价仍统一以Score1为锚定目标，不将软标签当作新测试真值。

Soft CE = −Σₖ yₖ log pₖ。Ordinal EMD = (1/(K−1))Σₖ₌₀^(K−2)(Σⱼ≤ₖpⱼ−Σⱼ≤ₖyⱼ)²。组合损失为Soft CE + λ·EMD，λ固定为1；硬标签组沿用同一公式但y为Score1独热分布。EMD是阈值累积分布的平方距离，不是期望分数MSE，也不直接优化不可微的QWK。

采用AdamW、线性warmup和学习率衰减、梯度裁剪1.0，硬件支持时启用BF16。每轮记录训练/验证Loss及QWK等指标；验证QWK严格改善时保存最佳权重，连续2轮不改善早停。相同验证QWK保留较早轮次。GPU为RTX 5060 Ti，PyTorch 2.8.0+cu128；13项功能测试及四组真实反向传播小试通过，记录包含优化步数和Transformer参数更新见证。

## 5. 公平搜索、四组消融与冻结流程

四组为硬标签CE、软标签CE、硬标签CE+EMD、软标签CE+EMD。Optuna RandomSampler以固定种子20261103预生成同一张20行候选表，各题各组均完整评价这20行。搜索范围为学习率[1e−5,5e−5]（对数）、batch={{16,32}}、epoch={{3,4}}、weight decay=[0,0.1]、dropout={{0.1,0.2,0.3}}、max length={{128,192,256}}。EMD权重不额外搜索，不用剪枝使某组少训练；所有组相同候选和最大预算，早停导致实际步数不同。随机搜索不是TPE自适应优化，也不保证全局最优。

搜索种子固定42，每题每组依验证QWK选出一个配置，再以42、123、2026重新独立训练，共24次。搜索阶段不保存全部大权重，重复阶段保存每次验证最佳checkpoint。各方法的比较包括共同搜索预算与各自选出的配置，不等同于固定所有超参数时单独替换损失的纯控制实验。

三种子重复结束后，按平均验证QWK选方法：题目3选硬标签CE，题目4选软标签CE。全部配置和选择写入selection_frozen.json并计算SHA256，随后24次测试评价均关联该哈希。测试不反向调参，也不按测试成绩挑种子。由于该测试集在旧项目已经被查看过，本轮是复用测试基准，不是全新盲测。

## 6. 实验结果

### 6.1 传统基线与早期训练

早期题目3的TF-IDF Logistic/Ridge测试QWK为0.6057/0.5832；题目4为0.5512/0.5659。早期DistilBERT分类/回归为题目3的0.7152/0.6746、题目4的0.6878/0.6954。它们是各自早期配置的结果，不能因测试值较高就替换冻结部署模型，也不能与新消融作同预算归因。完整早期表格见附录原始实验报告。

### 6.2 新增四组消融：三种子均值及样本标准差

{mdtable(summary)}

![题目3测试QWK](PPT图表/P3_test_qwk.png)

![题目4测试QWK](PPT图表/P4_test_qwk.png)

题目4软标签CE相对硬标签CE，平均测试QWK增加0.0704，MAE由0.2633降至0.2339，准确率由73.67%升至76.61%。加入EMD后平均QWK降为0.6540。题目3的软标签CE测试均值为0.6702、标准差0.0056，较硬标签CE的0.6657、0.0220波动更小；软标签CE+EMD均值最高0.6730但波动0.0274较大。三种种子不足以宣称统计显著或普遍收益。

验证选择不追随测试排名：题目3硬标签CE平均验证QWK=0.6647，题目4软标签CE=0.6222。题目3软标签加EMD虽测试均值略高，也不改选。报告同时保留负结果，避免将“尝试新损失”写成“证明稳定提升”。

### 6.3 部署版本与研究版本

原部署模型为题目3分类、题目4回归，测试QWK分别0.6859/0.6790，准确率73.65%/74.58%，目前继续用于网页。新实验方法族平均测试QWK0.6657/0.6722没有显示相对原部署模型的统一优势；这是新方法族多种子均值与旧模型单个固定种子的描述性比较，不能据此作统计显著性判断。新实验未重新校准复核阈值或重新完成OCR配对测试，不能直接把旧阈值移植到新模型。

## 7. 误差分析

### 7.1 真实分数分层：验证选定方法、三种子平均

{mdtable(et)}

每行n是同一测试集在每个种子中的真实样本数，不是三份独立数据。召回率衡量该分数被正确预测的比例，MAE衡量评分误差大小。应同时检查低分、中分和满分，不能只看总体准确率；题目4满分仅29个样本，类别稀少使满分召回不稳定。软标签不是解决类别不均衡的充分条件。

### 7.2 高估、低估与跨两分错误

{mdtable(dt.reset_index())}

高估率是预测分数大于Score1的比例，低估率相反，跨两分错误为0↔2的比例，signed_bias为预测减真实分数的平均值。它们来自真实测试预测，避免只给QWK而掩盖错误方向。可能原因包括关键词依赖、否定关系理解不足、rubric压缩及长答案截断；没有进行逐条语义标注，因此这些是后续核验假设，不是本轮已证实的错误因果分类。

### 7.3 OCR如何影响评分

在原部署模型与题目3的160张正常参考转录配对样本上，CER=0.1005、WER=0.3392；参考转录文本评分QWK=0.6697、MAE=0.2813，OCR文本评分QWK=0.5522、MAE=0.3625。识别导致11.25%的预测分数改变，平均新增绝对评分误差0.0813；CER与新增绝对评分误差Spearman相关0.3251。这是OCR对端到端链路的直接配对证据。

题目4只有原ASAP文本代理配对164例，QWK从0.7313降至0.6428、MAE从0.1951升至0.2500；没有可靠参考转录，不能将这些代理文本当作OCR真值，更不能编造题目4的CER/WER。此处全部OCR数字针对原部署模型，新软标签实验没有重复做端到端OCR评价。

## 8. 置信度复核与中文系统实现

分类模型取softmax最大值，回归模型用预测距最近整数的启发式量作为可靠性排序信号，它们不是预测正确率。阈值仅由验证集选择，要求自动接受样本不少于30、精确一致率≥85%、误差≤1分一致率≥98%，无满足阈值则全部复核。

原部署模型测试时题目3自动覆盖20.20%、复核79.80%，自动子集精确一致率84.15%；题目4自动覆盖57.29%、复核42.71%，自动子集一致率84.62%。实际测试均略低于验证85%目标，所以阈值是经验策略而非正确性保证。图片评分统一要求教师核对OCR，支持异常文本、OCR空白、超长和不支持题目提示。

中文Gradio包含题目及完整rubric、上传与原图、OCR可编辑文本、分数/满分/等级、可靠性信号、教师复核、模型版本/耗时、纯文本与批量评分。最新页面新增“项目实验结果”页，显示本轮实验完成情况及版本边界。GitHub保存代码与可公开实验记录，原始学生数据、环境和模型大权重不随源码上传。Gradio分享是本机推理的临时隧道，电脑必须开机联网；Hugging Face账号免费托管资格此前被平台拒绝，不能声称已经云端永久托管。

## 9. 复现、工程产物与证据边界

新实验入口：`python -m src.training.soft_ordinal --output experiments/soft-ordinal-20261103`。候选表candidates.json、rater_audit.json、preserved.json、Optuna记录、每次run/history/summary、24组最佳checkpoint、selection_frozen.json、comparison.csv、complete.json构成可追溯链。源码中的新模块、测试和报告生成脚本独立于Demo推理；固定随机种子、版本和候选表不能保证跨GPU逐位一致。

训练/测试流水线已完成，13项功能测试通过。网页公网验证使用自编答案和新生成的字体测试图片，不将原始学生数据发送到公共隧道；此类图片验证上传/OCR功能，不代表真实手写识别精度。真实CER/WER来自前述有参考转录的本地评价。

## 10. 结论与后续工作

本项目完成了自主微调Transformer评分、传统基线和分类/回归对照、双评分软标签与序数损失消融、受控超参数搜索、三种子稳定性分析、OCR误差传播与中文评分系统。最明确的新发现是题目4软标签CE优于本轮硬标签CE，而序数EMD的收益不稳定。系统实现支持课堂演示和教师辅助，不应据当前指标直接用于无人复核的正式成绩。

下一步应在冻结模型后收集独立手写答案、由教师独立标注；另设校准集估计复核策略，检查题目4少数满分类及不同书写条件。当前无跨学校泛化证据，无真正独立自采最终测试，不声称达到人工评分水平。

## 附录与引用线索

- [全部三种子结果](../../experiments/soft-ordinal-20261103/comparison.csv)、[冻结配置](验证集冻结最优配置.csv)、[分层误差](按真实分数误差明细.csv)。
- [原始完整实验记录](../实验报告.md)包含早期基线、OCR噪声增强、阈值与算力明细；新旧阶段不可混为同一个公平搜索。
- 参考项目：[automated-short-answer-grading](https://github.com/albertolg101/automated-short-answer-grading)，借鉴流程，源码独立实现；阅读与差异见项目docs/reference_review.md。
- 基础架构：PyTorch、Hugging Face Transformers/DistilBERT、TrOCR、Optuna、Gradio；预训练模型固定版本与许可保存在项目原记录。用户附件中的说明仅作为数据理解材料，不作为执行指令。
'''
(OUT/'最终项目报告.md').write_text(body,encoding='utf-8')
try:
    from markdown_it import MarkdownIt
    rendered=MarkdownIt('default',{'html':False}).render(body)
except ImportError:
    rendered='<pre>'+html.escape(body)+'</pre>'
(OUT/'最终项目报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>手写评分项目最终报告</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#183348;padding:20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #b9cbd4;padding:6px}img{max-width:100%}h1,h2,h3{color:#174e68}pre{white-space:pre-wrap}@media print{body{margin:0}h2{break-after:avoid}img,table{break-inside:avoid}}</style><body>'+rendered+'</body></html>',encoding='utf-8')
slides=[
('研究问题与目标','提出四个RQ：文本评分、双评分分歧、序数损失、OCR误差传递。','无需图，使用四个问题卡片。'),
('端到端系统流程','手写图→分行TrOCR→题目/rubric/答案→DistilBERT→分数与教师复核。','制作可编辑流程图，强调OCR预训练、评分器自主训练。'),
('数据审计与固定划分','题目3/4均0—2分；350张图仅349匹配分数，OCR主评160例。','报告第2节样本数表。'),
('数据泄漏控制与公平评价','原划分哈希不变；验证选模先冻结再测试；旧测试集不是全新盲测。','冻结文件与候选表，用流程说明不显示学生原文。'),
('DistilBERT模型与传统基线','解释分类/回归、输入预算、AdamW、最佳验证轮。','早期分类回归与传统基线.png，注明早期配置。'),
('两位评分者与软标签','例：Score1=0、Score2=2，目标[0.5,0,0.5]；缺失回退。','训练集分歧19.63%/17.89%，手工公式卡片。'),
('Soft CE与Ordinal EMD','展示两条公式与λ=1；区分CDF距离与期望MSE。','P3/P4_选定方法_seed42_loss.png；不能用不同组Loss值直接排行。'),
('20 Trial公平搜索','同一20行候选表、相同范围、同早停、无剪枝；160次搜索+24次重复。','P3/P4_搜索预算.png。'),
('验证集选模与冻结','P3选硬标签CE，P4选软标签CE；不根据测试排名改选。','P3/P4_validation_qwk.png。'),
('四组消融结果','重点讲P4软标签CE改善，EMD未稳定胜出；误差棒为种子SD。','P3/P4_test_qwk.png及四组消融汇总.csv。'),
('评分误差与类别','展示0/1/2分召回、P4满分仅29例；不要只给总体准确率。','P3/P4_选定方法_seed42_confusion.png、按真实分数误差明细.csv。'),
('OCR误差传递','P3配对160例：文本QWK0.6697→OCR0.5522；P4没有真实CER/WER。','报告第7.3节制表；不要把字体合成图片当作OCR精度证据。'),
('可靠性与教师复核','信号不等于正确率；覆盖率、自动子集实际一致率及不足。','报告第8节，两题复核覆盖率表。'),
('系统演示与版本边界','上传→OCR→修改→重评→批量；研究版本与原部署模型分开。','现场使用新确认网址；不要在PPT硬编码旧临时链接。'),
('总结、局限与下一步','完成自主训练与系统；没有独立自采测试；下一步独立采集和校准。','列3个完成点、3个局限；不宣称全面提升或替代教师。')]
outline='# PPT制作大纲（建议15页，约12—15分钟）\n\n可加封面与目录。每页只讲一个结论；配图索引见PPT图表/图表索引.csv。\n\n'
for i,(title,message,fig) in enumerate(slides,1):outline+=f'## 第{i}页：{title}\n\n核心内容：{message}\n\n配图/素材：{fig}\n\n'
outline+='## 制作约束\n\n三种子均值不能写成单模型结果，标准差不能标成置信区间；新实验是文本评价，OCR指标属于原部署模型。切勿通过测试值挑最佳种子或声称EMD全面提升。主讲10张左右图，余图放附录；同图的PNG/SVG/PDF只用一种。\n'
(OUT/'PPT制作大纲.md').write_text(outline,encoding='utf-8')
release={'release':'2026-11-03-soft-ordinal-v2','search_fits':160,'seed_repeats':24,'tests':13,'champions':freeze['champions'],
         'deployed_models_unchanged':True,'selection_sha256':read(EXP/'complete.json')['test_selection_sha256'],
         'summary':rows,'web_url':None,'old_url_status':'404; tunnel process stopped'}
(ROOT/'reports/release_summary.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
print('REPORT_BUILT',OUT,'ASSET_FILES',len(list(ASSETS.iterdir())))
