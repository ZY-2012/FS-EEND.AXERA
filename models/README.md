# models/

量化模型不放在本仓库。`streaming_step.axmodel`（AX650/NPU3，约 12.5 MB）与预编译的
C++ 可执行文件发布在 HuggingFace：

- 模型：https://huggingface.co/HY-2012/FS-EEND.AXERA/resolve/main/models/streaming_step.axmodel
- 预编译二进制：https://huggingface.co/HY-2012/FS-EEND.AXERA/tree/main/bin

```bash
mkdir -p models && cd models
wget https://huggingface.co/HY-2012/FS-EEND.AXERA/resolve/main/models/streaming_step.axmodel
```

也可以按 [model_convert/](../model_convert/README.md) 自行导出并量化。
