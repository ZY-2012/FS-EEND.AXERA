# models/

量化模型不放在本仓库，发布在 HuggingFace：
[HY-2012/FS-EEND.AXERA](https://huggingface.co/HY-2012/FS-EEND.AXERA)

上游为每个数据集单独训练了模型，`max_speakers` 不同 → 输出通道数不同：

| 变体 | 来源 checkpoint | max_speakers | 输出通道 | HF 路径 |
|---|---|---|---|---|
| `simu` | `ls_eend_1-8spk_16_25_avg_model` | 8 | 10 | `models/simu/streaming_step.axmodel` |
| `ami` | `ls_eend_ami_allspk_model` | 4 | 6 | `models/ami/streaming_step.axmodel` |

```bash
BASE=https://huggingface.co/HY-2012/FS-EEND.AXERA/resolve/main
mkdir -p models/simu models/ami
wget -P models/simu $BASE/models/simu/streaming_step.axmodel
wget -P models/ami  $BASE/models/ami/streaming_step.axmodel
```

预编译 C++ 可执行文件同样在 HF：
[`bin/ls_eend_ax650`](https://huggingface.co/HY-2012/FS-EEND.AXERA/tree/main/bin)

CALLHOME（7→9 通道）与 DIHARD2/3（10→12 通道）的 checkpoint 未发布量化产物，
但转换流程已支持，按 [model_convert/](../model_convert/README.md) 用
`LS_EEND_CONF` 指定对应 infer YAML 自行量化即可。
