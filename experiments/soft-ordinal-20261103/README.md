# 新一轮独立实验（运行中）

原划分、模型registry和部署权重的SHA256记录在preserved.json；流水线结束时强制复核。Gradio继续使用原模型，不自动晋升新模型。

四组：hard_ce、soft_ce、hard_ce_emd、soft_ce_emd。软目标为Score1/Score2独热分布的等权平均；缺失Score2回退Score1，非法分数拒绝处理。EMD为两个非平凡累积分布阈值上的均方距离，系数1。评价统一锚定Score1。

每题每组20个候选配置，所有组共享同一张Optuna RandomSampler候选表，搜索种子42、相同候选训练预算、相同早停规则、不剪枝。共160次搜索训练和24次三种子（42/123/2026）重复训练。搜索只保留指标和曲线，重复实验保存验证最佳权重。

所有重复完成后冻结selection_frozen.json及其SHA256，才读取测试集评价；不会以测试结果调参。旧项目已经评价过同一测试集，本次不能宣称全新盲测。

真实完成标记为complete.json。该文件出现前，报告和表格可能尚未产生或仅为中间结果，不能宣称全部实验完成。可从training.log、training.error.log及各次history.json查看真实进度。

可中断后恢复（请勿同时启动多个进程）：

E:/Intelligent-Handwritten-Grading/.venv/Scripts/python.exe -m src.training.soft_ordinal --output E:/Intelligent-Handwritten-Grading/experiments/soft-ordinal-20261103

结果：comparison.csv、stability.csv、report.md、figures/。图含三种子均值/标准差及单点、训练/验证Loss、验证QWK曲线、搜索预算曲线、测试混淆矩阵。字体/布局和PDF视觉QA待完整结果产生后检查，未声称已通过图表终审。
