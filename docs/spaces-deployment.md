# Hugging Face Spaces 部署

GitHub 保存项目源码与真实实验记录；Spaces 使用独立的最小推理目录，避免将 21 GB 本地环境、数据和历史 checkpoint 上传。

运行入口为根目录 `app.py`，监听 0.0.0.0:7860；`deploy/README-space.md` 提供 Gradio SDK 元数据，`deploy/requirements-spaces.txt` 固定 CPU 推理依赖。训练不在 Space 启动时执行。

部署目录须包含 `src/`、`demo/`、`configs/`、`models/registry.json` 及其指向的两个完整最优 checkpoint（包括 model.safetensors），另包含 `data/pretrained/trocr/` 的官方固定版本文件及第三方许可。评分模型和 OCR 都从本地资产加载，不需要商业模型 API。

将部署目录上传为 Gradio Space 后，等待 Hugging Face 构建为 RUNNING，再验证文本、图片、手动修改与批量接口。部署认证由正常 Hugging Face 本地登录读取，不在源码或 README 中保存访问令牌。

## 本次部署状态（2026-10-30）

已准备真实权重运行包，并在与云端相同的 CPU 入口上完成两题文本评分、多行 OCR、修改后重评和批量评分。Hugging Face 账号 zd2004 创建 Gradio Space 时，平台 API 返回 HTTP 402，明确要求 PRO 订阅。用户选择暂不开通，因此尚未创建或启动线上 Space，也没有可用的线上评分网址。部署包保留，账号条件满足后可继续上传。

状态记录：deploy/deployment_status.json。平台套餐说明：https://huggingface.co/pro 。
