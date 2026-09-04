# cpp/ — LS-EEND streaming diarization on AX650N

C++ 板端推理，wav → RTTM。推理逻辑与 `../python/ls_eend_sdk` 逐项对齐（同一前端、
同样的 warmup 门控、同样的 FP32 主机端累加、同样的后处理）。

## 目录

| 文件 | 作用 |
|---|---|
| `ls_eend_ax.cpp` | 主程序（命令行解析 + 流程编排） |
| `src/audio_reader.cpp` | WAV 读取 + 线性重采样到 8 kHz |
| `src/feature_extractor.cpp` | `logmel23_cummn` 前端（KissFFT），逐点对齐 librosa |
| `src/ax_engine_runner.cpp` | AX Engine 命名多路 IO 封装（14 入 / 12 出） |
| `src/streaming_diarizer.cpp` | 流式循环 + warmup 门控 + FP32 累加 + RTTM |
| `tools/dump_features.cpp` | 只导特征，用于和 Python 对分（不依赖 BSP） |
| `third_party/kissfft/` | FFT（随仓提交） |

## 工具链

不随仓提交，一键下载：

```bash
bash download_toolchains.sh
```

会拉到 `cpp/toolchains/`：

| 用途 | 来源 |
|---|---|
| aarch64 交叉编译器 | `gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu`（ARM 官方 tar.xz） |
| AX650N BSP SDK | `git clone --depth=1 https://github.com/AXERA-TECH/ax650n_bsp_sdk` |

已有本地副本时用环境变量指过去即可：

```bash
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
bash build_ax650.sh
```

## 编译

```bash
bash build_ax650.sh          # -> cpp/bin/ls_eend_ax650
```

## 运行（板端）

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/ls_eend_ax650 \
    --model models/streaming_step.axmodel \
    --wav   samples/mix_0000176.wav \
    --rttm  out.rttm \
    --max-speakers 4
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | `streaming_step.axmodel` |
| `--wav` | 必填 | 输入音频，任意采样率（内部重采样到 8 kHz） |
| `--rttm` | `<wav>.rttm` | 输出 RTTM |
| `--max-speakers` | 8 | 保留 8 个说话人通道中的前 N 个 |
| `--threshold` | 0.5 | sigmoid 活动判决阈值 |
| `--median` | 11 | 中值滤波长度（11 与上游一致） |

## 板端实测（AX650N，`mix_0000176.wav`，192 s / 1921 帧）

| 指标 | C++ | Python |
|---|---|---|
| ms/帧 | **2.50** | 7.29 |
| RTF | **0.0250** | 0.0726 |
| DER（collar 0.25） | 2.63% | 1.95% |
| 段数 | 57 | 55 |

C++ 比 Python 快约 2.9 倍（主机端 FP32 累加在 numpy 里开销较大）。

两者互相分歧 1.01%（把 Python 输出当参考、collar 0 算 DER）。残差来自前端 2.3e-05
的浮点差异经 1900 步递推后，在少数接近阈值的帧上翻转判决——不是逻辑差异。

## 前端对分

前端可以脱离 BSP 单独验证：

```bash
g++ -O2 -std=c++11 -I include -I third_party/kissfft \
    tools/dump_features.cpp src/audio_reader.cpp src/feature_extractor.cpp \
    third_party/kissfft/kiss_fft.c -lm -o dump_features
./dump_features input.wav features.bin
```

```python
import numpy as np, sys
sys.path.insert(0, '../python')
from ls_eend_sdk.feature import wav_to_features
py, _ = wav_to_features('input.wav')
cpp = np.fromfile('features.bin', dtype=np.float32).reshape(-1, 345)
print('cosine', float((py.ravel() @ cpp.ravel()) /
                      (np.linalg.norm(py) * np.linalg.norm(cpp))))
```

实测 `cosine 0.99999988`、`max diff 2.3e-05`。

前端有四个容易踩错、必须与 librosa 一致的点：

1. **Slaney mel 尺度**（`librosa.filters.mel` 默认 `htk=False`），不是
   `2595*log10(1+f/700)`。
2. **`norm='slaney'`**：每个三角滤波器要乘 `2/(f[i+2]-f[i])`。
3. 三角滤波器建在**连续频率**上，不能先把边界 floor 到 FFT bin。
4. **周期 Hann 窗**（`fftbins=True`，分母是 `win_length` 而不是 `win_length-1`）。

早期版本用了 HTK 尺度 + 无归一化 + floor 到 bin 的写法，板端 DER 33.6%。

## 已知的编译坑

- kissfft 是 C 源码，需要**独立的 C 目标**（`project(... LANGUAGES C CXX)` +
  单独 `add_library`），否则纯 C++ 工程会静默跳过它，报
  `undefined reference to kiss_fft_alloc`。
- 交叉编译要**同时**传 `CMAKE_C_COMPILER` 和 `CMAKE_CXX_COMPILER`。只传 C++ 的会用
  宿主 gcc 编 kissfft，链接时报
  `Relocations in generic ELF (EM: 62)`。
- `AX_NPU_*` 符号在 `libax_interpreter`，不在 `libax_engine`——三个库
  （`ax_engine ax_interpreter ax_sys`）都要链。
- `AX_SYS_Init()` 必须在 `AX_ENGINE_Init()` 之前调用，否则后者返回错误。
