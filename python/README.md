# python/ — LS-EEND streaming diarization SDK

Python 板端推理（axengine），wav → RTTM。

## 安装

```bash
pip install -r requirements.txt      # axengine 板端镜像已自带
```

## 用法

```bash
python example.py \
    --model ../models/simu/streaming_step.axmodel \
    --wav   ../samples/mix_0000176.wav \
    --rttm  out.rttm \
    --max-speakers 4
```

也可以传 `.onnx` 在宿主机上跑（需要 `onnxruntime`），用于和板端对分。

作为库调用：

```python
from ls_eend_sdk import diarize

result = diarize('meeting.wav', 'models/simu/streaming_step.axmodel', max_speakers=4)
print(result['rttm'], result['speakers'], result['rtf'])
for start, end, spk in result['segments']:
    print(f'{start:.2f}-{end:.2f} speaker_{spk}')
```

逐帧流式接口（用于真实实时场景）：

```python
from ls_eend_sdk import StreamingDiarizer, extract_features, load_audio

runner = StreamingDiarizer('models/simu/streaming_step.axmodel')
audio, _ = load_audio('meeting.wav')
for frame in extract_features(audio):
    logits = runner.step(frame)     # 前 9 帧返回 None（卷积 warmup）
    if logits is not None:
        ...                         # (10,) logits：ch0 静音，ch1..8 说话人，ch9 非说话人
```

## 模块

| 文件 | 作用 |
|---|---|
| `feature.py` | `logmel23_cummn` 前端（8 kHz、23 mel、±7 拼接、10 倍下采样） |
| `session.py` | 流式循环：warmup 门控 + FP32 主机端累加 |
| `postprocess.py` | sigmoid → 阈值 → 11 帧中值滤波 → RTTM |
| `diarize.py` | 端到端入口 |

## 两个必须遵守的约定

导出的图不是自包含的，`session.py` 里实现了这两点，自行改写时不能省：

1. **FP32 主机端累加**：图只返回有界的单帧增量 `enc{i}_inc` / `dec{i}_inc`，
   主机侧用 FP32 维护 `mean += (inc - mean) / t`。把量化后的状态直接喂回会让误差
   沿递推累积（DER 1.95% → 7.4%）。
2. **Warmup 门控**：上游 `StreamingConv1d` 前 9 帧不输出，原生循环此时完全不调用
   decoder。所以前 9 帧喂真实特征让 encoder 状态前进，但丢弃 `pred` 且保持 decoder
   状态不变（漏掉这步 DER 会到 71.8%）。

## 板端实测（AX650N，192 s / 1921 帧）

| 指标 | 值 |
|---|---|
| ms/帧 | 7.29 |
| RTF | 0.0726 |
| DER（collar 0.25） | 1.95%（miss 1.39%、FA 0.56%、confusion 0%） |

同一模型的 C++ 版本快约 2.6 倍且 DER 相同，见 [../cpp/README.md](../cpp/README.md)。

## 输出时间轴

每帧 0.1 s。前 9 帧不产生输出，末尾 0.9 s 也不输出——原生 flush 是把零 *embedding*
直接推进输出卷积、绕过 encoder，融合的单帧图表达不了这条路径。
