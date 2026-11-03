# 最新网页与部署状态（2026-11-03）

已验证的新地址：[中文手写评分网页](https://766367126f380a0b07.gradio.live)。

原地址 https://3f431a2aa4e22dab2c.gradio.live 已返回404，分享进程已停止，不能声称原链接已更新。新链接由同一套真实评分资产启动，新增“项目实验结果”页面及project_status接口，接口确认版本2026-11-03-soft-ordinal-v2、160次搜索、24次重复、8组汇总。

公网完成两题真实模型评分、合成字体图片上传/OCR/评分、修改后重评、批量评分5项检查；另外浏览器验证最新页、上传、重评、批量及HTML报告。更新后13项功能测试通过。所有公开测试采用自编答案和新生成的字体图片，没有发送学生原始数据。合成图片只作功能验收。

当前评分继续使用原部署模型与原复核阈值；新研究模型不自动晋升。原划分、registry和部署权重SHA256再次核验不变。

这是本机推理的Gradio临时隧道，电脑需保持开机联网。当前版本提示链接约1周有效，停止/重启可能改变地址。没有永久云托管或Hugging Face已部署的承诺。

证据：reports/final-20261103/public-endpoint-tests.json、browser-release-tests.json、functional-tests.xml及delivery-verification.json。
