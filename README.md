# FS-EEND.AXERA

LS-EEND（FS-EEND 的在线版本）说话人日志（speaker diarization）模型的 Axera AX650N
部署工程：输入音频，输出 RTTM 说话人时序标签。

原生**逐帧流式**推理路径完整迁移到 NPU，量化后与原生 PyTorch 对齐：

### 仿真样本（`test_samples/mix_0000176.wav`，192 s / 1921 帧 / 4 人）

| 路径 | vs 原生 cosine | DER（±0.25 s，上游口径） | DER（±0.125 s） | ms/帧 | RTF |
|---|---|---|---|---|---|
| 原生 PyTorch 逐帧 | — | **0.4411%** | 1.2156% | — | — |
| FP32 ONNX 流式循环 | **1.0000000** | 0.8672% | 1.6489% | — | — |
| **AX650N Python** | **0.9985337** | **1.1214%** | 1.9512% | 7.29 | 0.073 |
| **AX650N C++** | 0.9985337 | **1.1214%** | 1.9512% | **2.79** | **0.0279** |

两条板端路径 confusion 均为 **0%**（说话人指派完全正确），DER 完全相同。

> 口径说明：上游 `metrics.py` 用 `DiarizationErrorRate(collar=50)`，而其 `Segment`
> 单位是 10 ms 帧，注释写明 "250ms tolerance" —— 等效 pyannote 的 `collar=0.5`
> （±0.25 s，NIST 惯例）。表中两列都给出，便于对照。

### AMI Eval（真实 16 kHz 会议，`ls_eend_ami_allspk_model.ckpt`）

参考 RTTM 由 AMI ASR 切片文件名的时间戳重建，音频为官方 Mix-Headset 混音；
协议按上游 README：**不用 collar、不用中值滤波**，`max_speakers=4`。

| meeting | 帧数 | 原生 FP32 | 板端 U16/S8 | Δ |
|---|---|---|---|---|
| ES2004a | 10493 | 19.61% | 19.75% | +0.14 |
| IS1009a | 8388 | 27.33% | 28.09% | +0.76 |
| EN2002d | 22098 | 27.59% | 29.97% | +2.38 |
| EN2002a | 21427 | 27.60% | 34.70% | +7.10 |
| **加权（4 场）** | | **26.50%** | **30.18%** | **+3.68** |

板端 2.2 ms/帧。上游 README 的 AMI Eval 是 20.76%（完整 16 场、官方 NXT 标注）。

- [x] 模型导出 + 导出对分（`model_convert/`）
- [x] 量化校准数据生成 + Pulsar2 量化（`model_convert/`）
- [x] Python 板端推理（`python/`）
- [x] C++ 板端推理（`cpp/`）

量化模型与预编译二进制在 HuggingFace：
[HY-2012/FS-EEND.AXERA](https://huggingface.co/HY-2012/FS-EEND.AXERA)

上游模型：[Audio-WestlakeU/FS-EEND](https://github.com/Audio-WestlakeU/FS-EEND)

## 支持模型

上游为每个数据集单独训练了一个模型，`max_speakers` 不同导致输出通道数不同。
转换流程对四个 checkpoint 都适用（用 `LS_EEND_CONF` 指定对应 infer YAML）：

| checkpoint | infer YAML | max_speakers | 输出通道 | 已验证 |
|---|---|---|---|---|
| `ls_eend_1-8spk_16_25_avg_model` | `..._infer.yaml` | 8 | 10 | ✅ 量化 + 板端 |
| `ls_eend_ami_allspk_model` | `..._ami_infer.yaml` | 4 | 6 | ✅ 量化 + 板端 |
| `ls_eend_ch_allspk_model` | `..._callhome_infer.yaml` | 7 | 9 | 脚本支持，未测 |
| `ls_eend_dih2/dih3_allspk_model` | `..._dihard2/3_infer.yaml` | 10 | 12 | 脚本支持，未测 |

模型接口：`feat [1,1,345]` + 10 路状态 → `pred [1,1,C]` + 11 路状态，axmodel 约 12.5 MB，
U16 激活 / S8 权重。8 kHz 输入，每帧 0.1 s；通道布局为 ch0 静音、ch1..C-2 说话人、
ch(C-1) 非说话人。

## 目录结构

```
FS-EEND.AXERA/
├── model_convert/       # 导出 ONNX → 对分 → 校准数据 → Pulsar2 量化
├── python/              # Python 板端推理 SDK（axengine）
├── cpp/                 # C++ 板端推理源码 + 交叉编译脚本
├── models/              # 仅 README，模型从 HuggingFace 下载
└── requirements.txt
```

## 快速开始（板端推理）

```bash
# 1) 取量化模型
mkdir -p models/simu && cd models/simu
wget https://huggingface.co/HY-2012/FS-EEND.AXERA/resolve/main/models/simu/streaming_step.axmodel
cd ..

# 2) Python
pip install -r requirements.txt
python python/example.py --model models/simu/streaming_step.axmodel \
                        --wav your.wav --max-speakers 4

# 3) C++
bash cpp/download_toolchains.sh      # gcc 交叉编译器 + AX650N BSP
bash cpp/build_ax650.sh              # -> cpp/bin/ls_eend_ax650
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./cpp/bin/ls_eend_ax650 --model models/simu/streaming_step.axmodel \
                        --wav your.wav --max-speakers 4
```

详见 [python/README.md](python/README.md) 与 [cpp/README.md](cpp/README.md)。

## 模型转换

完整步骤见 [model_convert/README.md](model_convert/README.md)：

```bash
export LS_EEND_REPO=/path/to/FS-EEND/LS-EEND
export LS_EEND_CKPT=/path/to/ls_eend_1-8spk_16_25_avg_model.ckpt

cd model_convert
python export_streaming_step.py        # ① 导出 ONNX + 单步对分
python verify_mean_form.py             # ② 验证与上游等价
python generate_real_calibration.py    # ③ 校准数据
bash pack_calib.sh
bash compile.sh                        # ④ Pulsar2 量化编译
```

工具链本地已有时不必重复下载，见 [cpp/README.md](cpp/README.md) 的环境变量说明。

## 关键实现要点

把这个模型跑对，有三个坑必须处理。三者都在 `python/ls_eend_sdk` 与 `cpp/` 中实现，
自行改写推理循环时不能省。

### ① Warmup 门控（漏掉 → DER 71.8%）

上游 `StreamingConv1d` 缓冲 19 帧、直到第 10 次调用才输出，所以原生循环在前 9 帧
**完全不调用 decoder**。推理循环必须：前 9 帧照常喂真实特征（encoder 状态要前进），
但丢弃 `pred` 且保持 decoder 状态不变。

### ② 校准数据要覆盖整段录音（否则 DER 7.4%）

retention 状态是累积量。原始形式下状态会随时间增长到 ±45k，只用开头几十帧校准会让
U16 MinMax 截断之后的所有帧。必须在全程均匀采样。

### ③ 有界均值形式 + FP32 主机端累加（两者缺一 → DER 7.4~9.2%）

导出时把 retention 状态从上游的 `kv_t = K_t/√t`（按 `√t` 增长到 ±45k）换成运行均值
`M_t = K_t/t`（有界在 ±1200）。这是**严格等价**而非近似：本 checkpoint
`decay = log([1,1,1,1])` 恰为 1，且 retention 输出只喂给
`group_norm`（`LayerNorm(elementwise_affine=False)`），对正标量因子不变。

但仅换形式反而更差（DER 9.17%）：`1/t` 很小时量化域里的增量会舍入成 0，均值停止更新。
所以图改为只输出**有界的单帧增量** `enc{i}_inc` / `dec{i}_inc`，由主机侧用 FP32 维护
`mean += (inc - mean)/t`。这样量化误差只影响当帧、不再沿递推累积。

## 已知限制

- **`decay == 1` 是均值形式成立的前提。** 换成 per-head decay < 1 的 checkpoint 必须
  恢复 `scale_t = scale_{t-1}*decay + 1` 的状态。
- **尾部 0.9 s 不输出。** 原生 flush 把零 *embedding* 直接推进输出卷积、绕过 encoder，
  融合的单帧图表达不了。
- **前端有状态。** `logmel23_cummn` 用累积均值归一化，跨调用不是无状态的。
- **模型是 8 kHz 的**：上游配置 `sample_rate: 8000`，`datasets/feature.py` 里 `sr = 8000`
  硬编码，10 个 LS-EEND config 全部用 `logmel23_cummn`。上游 README 也明确写了
  「AMI, DIHARD2 and DIHARD3 data are down-sampled to 8 kHz」。所以降到 8 kHz 是既定
  路径，不是妥协。本 SDK 的两条路径都会自动带限重采样（Python 用 librosa
  `kaiser_best`，C++ 用同参数的 Kaiser 窗 sinc）。
  注意上游 `extract_fbank()` **不做重采样**、直接丢弃 `sf.read` 返回的采样率，
  所以拿 16 kHz 文件喂原生脚本会静默得到错误特征且时间轴差 2 倍。
- **量化退化随录音长度增长**，且几乎全部落在 confusion（说话人指派）：
  8~10k 帧（14~17 min）时 Δconfusion 仅 +0.5~0.8 pp，22k 帧（36 min）时可达
  +1.4~5.9 pp。怀疑主因是 `inv_count = 1/t` 在 2 万帧时降到 5e-5，已低于校准下界，
  且 U16 单一尺度难以同时覆盖 1.0 与 5e-5 五个数量级。**长会议场景建议用同等长度的
  录音做校准。**
- **每个 checkpoint 的输出通道数不同**，由其 infer YAML 的 `max_speakers` 决定：
  simu 8→10、AMI 4→6、CALLHOME 7→9、DIHARD2/3 10→12。转换脚本用 `LS_EEND_CONF`
  切换；两个 SDK 都在加载时从模型读取通道数，不写死。

## License

[Apache-2.0](LICENSE)。上游 FS-EEND 与 LS-EEND 权重的许可以其原仓库为准。
