# 数据与权重准备

原始 ASAP-SAS 和 Handwritten-ASAP-SAS 数据包由项目使用者按其许可自行获取，原始数据及学生转录不在 Git 中再分发。手写数据原始说明注明仅限非商业研究。数据包内的原始许可、作者信息与引用说明应随本地数据保留。

将两份 ZIP 放在本机任意位置，再按仓库 README 的 audit_data.py 命令传入路径。脚本会在本项目内复制、审计并建立固定划分。data/processed 中已提供原实验的划分哈希与 OCR 开发/留出规则，便于核对复现。

最优评分模型所在路径见 models/registry.json。Git 版本含模型配置与 tokenizer，但不含 model.safetensors；请从项目所有者的本地完整项目取得对应权重，或执行训练和评价命令重新生成。预训练 DistilBERT 和 TrOCR 可通过 scripts/download_models.ps1 从固定官方版本下载。

独立自采测试接口见 self_collected/README.md。当前没有真实小组自采最终测试样本，不编造相关指标。
