# 模型转换（LS-EEND streaming）

从上游 FS-EEND / LS-EEND 权重导出**逐帧流式** ONNX，并用 Pulsar2 量化编译成
AX650/NPU3 的 `streaming_step.axmodel`。

## 环境

```bash
conda create -n ls_eend python=3.10 && conda activate ls_eend
pip install -r ../requirements.txt

# 上游工程（提供模型定义与特征提取）
git clone https://github.com/Audio-WestlakeU/FS-EEND.git
```

权重 `ls_eend_1-8spk_16_25_avg_model.ckpt` 从上游仓库的发布页获取。

路径通过环境变量指定，脚本本身不含任何机器相关路径：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `LS_EEND_REPO` | 上游 `FS-EEND/LS-EEND` 目录 | `../../FS-EEND/LS-EEND` |
| `LS_EEND_CKPT` | PyTorch 权重 | `$LS_EEND_REPO/../ls_eend_1-8spk_16_25_avg_model.ckpt` |
| `LS_EEND_WAV` | 校准与验证用录音 | `$LS_EEND_REPO/test_samples/mix_0000176.wav` |
| `LS_EEND_OUT` | 产物根目录 | 本目录 |

```bash
export LS_EEND_REPO=/path/to/FS-EEND/LS-EEND
export LS_EEND_CKPT=/path/to/ls_eend_1-8spk_16_25_avg_model.ckpt
```

## 步骤

### ① 导出 ONNX

```bash
python export_streaming_step.py
```

产出 `export/streaming_step.onnx`（14 输入 / 12 输出，全 FP32 边界）+
`export/model_meta.json`，并自动做 torch↔ONNX 单步对分（min cosine ≈ 0.9999999）。

这一步把上游 `MultiScaleRetention.recurrent_forward` 换成**有界均值形式**
（补丁只作用于导出进程，不改上游源码）。原因：上游状态 `kv_t = K_t/√t` 会随时间
按 `√t` 增长到 ±45k，单一 U16 量化尺度无法覆盖。改存运行均值 `M_t = K_t/t` 后状态
有界在 ±1200。

这不是近似——本 checkpoint 的 `decay = log([1,1,1,1])` 恰好为 1，且 retention 输出
只喂给 `group_norm`（`LayerNorm(eps=1e-6, elementwise_affine=False)`），对正标量因子
不变，所以丢掉 `√t` 增益后层输出完全一致。

### ② 验证均值形式与上游等价

```bash
python verify_mean_form.py
```

全录音比对，应输出 `cosine vs upstream 1.0000001`、`maxdiff 0.0003`，并打印各状态的
幅度上界（确认有界）。

### ③ 生成量化校准数据

```bash
python generate_real_calibration.py
bash pack_calib.sh
```

在**整段录音**上均匀采样 64 帧（`np.linspace(DELAY, total-1, 64)`）的真实状态。
必须覆盖全程：状态是累积量，只用开头几十帧校准会让 U16 MinMax 截断后面所有帧
（实测 DER 从 1.95% 恶化到 7.4%）。

产出 `calib_data/<input>/*.npy` 与打包后的 `calib_data/<input>.tar.gz`（14 个）。

### ④ Pulsar2 量化编译

```bash
bash compile.sh          # 等价于 pulsar2 build --config pulsar2_config.json
```

产出 `compile/streaming_step.axmodel`（约 12.5 MB）。

### ⑤ 全链路 FP32 对分（可选但建议）

```bash
# 先用上游 streaming_infer_dia.py 存一份逐帧参考 logits 到
# $LS_EEND_OUT/torch_frame_streaming_pred.npy，然后：
python validate_streaming_full_onnx.py
```

应输出 `cosine 1.0000000`、`frames with maxdiff>0.5: 0/1912`。

## 量化配置

`pulsar2_config.json`：

| 项 | 值 |
|---|---|
| `target_hardware` / `npu_mode` | `AX650` / `NPU3` |
| 激活 `data_type` / `output_data_type` | **U16** |
| 权重 `weight_data_type` | **S8** |
| `calibration_method` | `MinMax`（状态是累积量，不能用 Percentile 截断） |
| `calibration_size` | 64（每个输入一个 tar.gz） |
| `highest_mix_precision` | `false` |
| `precision_analysis_method` | `EndToEnd` |
| `compiler.check` | `0` |

`layer_configs` 只有一条全局 `DEFAULT` 规则，无 per-layer 覆盖。

### 试过但被否的精度方案

| 方案 | 结果 |
|---|---|
| 全局 `weight_data_type: S16` | `TileFailException("AxQuantizedConv, not enough values to unpack (expected 4, got 2)")` |
| 仅 decoder FFN 层（`op_22/23/24/107/108:onnx.FullyConnected`）S16 权重 | 该层 cosine 0.870→0.891，但板端 DER 反而变差（7.61% vs 7.42%） |
| state 边界设 `FP32` | AX650 无法 tile FP32 的 `AxLayerNorm` |
| 任何 `FP16` | `sepc_type FP16 not in STRING_NUMBER_MAP`，AX650 算子规格不支持 |

`layer_configs` 的合法字段只有 `layerName / opType / startTensorNames /
endTensorNames / dataType / weightDataType / layerNames / opTypes /
outputDataType`——`tensor_name` 不是合法字段。层名必须取自
`compile/work/frontend/optimized.onnx`（onnxsim 之后的名字与原始 ONNX 不同）。

当前最差层仍是 decoder FFN（`op_22` cosine 0.868、`op_24` 0.896），但端到端结果说明
它不是瓶颈。

## 调用方必须实现的两件事

导出的图不是自包含的，推理循环必须做到（参考 `../python/ls_eend_sdk/session.py`）：

1. **FP32 主机端累加**：图只返回有界的单帧增量 `enc{i}_inc` / `dec{i}_inc`，
   主机侧用 FP32 维护 `mean += (inc - mean) / t`。若把量化后的状态直接喂回，
   误差会沿递推累积。
2. **Warmup 门控**：上游 `StreamingConv1d` 前 9 帧不输出，原生循环此时**完全不调用
   decoder**。所以前 9 帧要喂真实特征（encoder 状态必须前进）但丢弃 `pred`、
   并保持 decoder 状态不变。漏掉这步 DER 会到 71.8%。
