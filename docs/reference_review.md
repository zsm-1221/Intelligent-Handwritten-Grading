# 参考仓库阅读记录与独立设计

参考：https://github.com/albertolg101/automated-short-answer-grading （2026-10-30 阅读 main 的 README.md、src/asag/data.py、train_transformer.py、evaluate.py、scoring.py、metrics.py）。只参考流程，未克隆或复制仓库。

- 数据：原仓库从远端 AERA 镜像读取科学题，使用 Score1，Score2 衡量人工一致性。本项目仅使用用户提供压缩包的 Prompt 3、4，题目、完整 rubric 从原始 DOCX 提取，核实满分为 2；公开测试文本按 `(EssaySet, Id)` 与官方 solution 匹配。
- 输入：借鉴 question + rubric + answer。独立实现固定上下文预算、答案独立截断；完整 rubric 在网页显示，训练采用存档的紧凑 rubric 与原问题。
- 训练：参考仓库共享一个四分类模型、固定参数、保存最终轮。本项目每题独立模型，对比三分类和回归，使用真实反传、AdamW、warmup/linear scheduler、early stopping，按 validation QWK 保存最佳模型；搜索和复现实验只读训练/验证集。
- 评价：参考仓库报告 QWK 和人工一致性。本项目追加 MAE、RMSE、Accuracy、混淆矩阵、逐样本预测及随机种子统计；QWK 在实际标签分布下计算，退化时不伪造 1.0。
- 复核：借鉴验证集选择复核阈值。原方法的相邻类别概率和，在 0～2 分的中间分数上总为 1，不适合直接沿用。本项目用分类最大 softmax／回归取整距离启发式信号作为排序信号，按验证集实际一致性选择阈值；达不到预设标准则全部复核，不使用未经证实的默认阈值。信号不称作正确率。
- OCR：独立添加 TrOCR、多行切分、参考转录核验、CER/WER，以及人工文本与 OCR 文本的评分差异分析。

库文档：https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/distilbert 、https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/trocr 、https://optuna.readthedocs.io/en/v4.5.0/ 、https://www.gradio.app/docs/gradio/blocks 。
