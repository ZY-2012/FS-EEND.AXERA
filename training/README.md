# LS-EEND 中英联合 finetune 训练包

把上游 FS-EEND 的 4 说话人模型从「中文 AliMeeting 上失能」（DER 64.01%）finetune 到
**21.14%**（接近 3D-Speaker +overlap 的 18.51%），且英文 AMI 不退化（26.89%）。
本目录包含完整可复现链路：数据下载 → 数据制作 → 训练 → 推理 → 评测。

**指标对比见 [RESULTS.md](RESULTS.md)。**

---

## 0. 环境

- Python 3.9，PyTorch **2.0.1+cu118**，pytorch-lightning **1.8.6**（上游 README 指定组合）
- 其它依赖：`hyperpyyaml soundfile librosa h5py`
- 上游代码：`git clone https://github.com/Audio-WestlakeU/FS-EEND.git`，
  训练脚本用本目录 `train/train_dia_fintun_real_patched.py` 替换
  `FS-EEND/LS-EEND/train_dia_fintun_real.py`（5 处必要修补，见第 4 节）
- 初始权重：上游 simu checkpoint `ls_eend_1-8spk_16_25_avg_model.ckpt`（Google Drive，
  见上游 README）
- 目录约定（下文脚本中的硬编码路径，按需修改）：
  - 训练机数据盘：`/root/autodl-tmp/data/`
  - kaldi 风格数据目录：`{wav.scp, segments, utt2spk, spk2utt, reco2dur, reco2num_spk, utt2timestamp}`

```bash
conda create -n ls_eend python=3.9 -y && conda activate ls_eend
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==1.8.6 hyperpyyaml soundfile librosa h5py
```

---

## 1. 数据下载

### 数据总览（训练 + 测试 + 参考）

| 数据集 | 用途 | 规模 | 下载地址 | 标注 | 备注 |
|---|---|---|---|---|---|
| AliMeeting far **训练集** | 训练（中文会议） | 78.6 GB（212 场） | [OpenSLR 119](https://www.openslr.org/119/) → 阿里云 OSS 直链（见 1.1） | 官方 TextGrid | 8 麦克风阵列 16 kHz，多通道声学增强的数据来源；OSS 约 10 MB/s |
| AliMeeting **Eval** | 测试（中文会议） | 3.7 GB + TextGrid | 同上 | 官方 TextGrid | 取 ch0 降混；评测 4 场（R8001_M8004 / R8003_M8001 / R8007_M8010 / R8007_M8011） |
| AMI **训练集** | 训练（英文会议） | 137 场 × ~200 MB | [AMI 镜像](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/) `amicorpus/{id}/audio/{id}.Mix-Headset.wav` | segments XML（[ami_public_manual_1.6.2.zip](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip)，22 MB） | ⚠️ 镜像限速 0.4~3.8 KB/s，**先下完再处理**（曾因边下边处理截断 37% 数据） |
| AMI **dev** | 测试（英文会议） | 16 场 | 同上 | NXT words 标注（`{id}.{spk}.words.xml`，同一 zip 内） | 报告用 12 场（ES2004b/IS1009b/TS3003b 留作调参，TS3003d 无音频） |
| VoxConverse **test** | 测试（近场媒体，1~21 人） | 4.3 GB（232 场） | [官方仓库](https://github.com/joonson/voxconverse)（zip 直链见其 README）；镜像 [diarizers-community/voxconverse](https://huggingface.co/datasets/diarizers-community/voxconverse)（parquet） | 官方 `test/*.rttm`（v0.3） | 牛津直链限速 ~190 KB/s，HF 镜像快一个量级 |
| 仿真样本 `mix_0000176.wav` | 冒烟测试 | 192 s / 4 人 | [FS-EEND 仓库](https://github.com/Audio-WestlakeU/FS-EEND) 自带 `test_samples/` | 自带 rttm | LS-EEND 同域，仅用于部署对分，不代表真实性能 |

划分规则：AMI 训练/测试按上游 split（ES2004/IS1009/TS3003/EN2002 四组留作 dev）；
AliMeeting 训练集中另留 10 场 held-out 作调参验证（按会议切分，防止同会议泄漏）。



### 1.1 AliMeeting（中文，远场会议，8 麦克风阵列）

OpenSLR 119：https://www.openslr.org/119/ （阿里云 OSS 直链，10 MB/s 量级）

| 文件 | 大小 | 用途 |
|---|---|---|
| `Train_Ali_far.tar.gz` | 78.6 GB | 训练集（212 场，含 8 通道音频 + TextGrid 标注） |
| `Eval_Ali.tar.gz` | 3.7 GB | 评测集（本包只用于评测，不进训练） |

```bash
aria2c -x 16 -s 16 -k 5M \
  -o Train_Ali_far.tar.gz \
  "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz"
tar -xzf Train_Ali_far.tar.gz
```

### 1.2 AMI（英文，远场会议，4 人）

AMI 语料镜像：https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/

- 音频：`{meeting}/audio/{meeting}.Mix-Headset.wav`（16 kHz 单声道混音）
- 标注：`https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip`
  （22 MB，segments XML）
- **训练/验证划分**：LS-EEND 上游把 `ES2004/IS1009/TS3003/EN2002` 四组（16 场）留作
  dev/eval，其余全部用于训练（约 137 场）

```bash
# 会议清单
curl -s https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ \
  | grep -oE 'href="[A-Z]{2}[0-9]{4}[a-d]/"' | sed 's|href="||;s|/"||' \
  | grep -vE '^(ES2004|IS1009|TS3003|EN2002)' > ami_train_meetings.txt

# 并行下载 Mix-Headset（xargs -P 16）
cat ami_train_meetings.txt | xargs -P 16 -I{} bash -c \
  'curl -s --retry 5 -o {}.Mix-Headset.wav \
   https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{}/audio/{}.Mix-Headset.wav'

# 标注
curl -s -o ami_public_manual.zip \
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip -o ami_public_manual.zip
```

> ⚠️ AMI 镜像限速严重（实测 0.4~3.8 KB/s，137 场需数小时~数天）。务必**先下完再跑
> 数据制作**，并逐场校验时长——我们曾因边下边处理导致 53/153 场被截断（训练损失 37%
> 英文数据），是英文侧退化的主因。

### 1.3（可选）VoxConverse / 其它评测集

评测集下载与复现见仓库主 README 或
[对比报告](../../LS-EEND_vs_3D-Speaker_对比报告.md)第 5.5 节（HF 镜像
`diarizers-community/voxconverse` 速度远快于牛津服务器）。

---

## 2. 数据制作（kaldi 风格目录）

统一格式：8 kHz 单声道 wav + 说话人级标注。**每次转换强制校验
「音频时长 ≥ 标注跨度 × 95%」，不合格的会议直接剔除**，禁止静默截断使用。

```bash
# 2.1 AliMeeting：TextGrid 说话人 tier → segments/utt2spk/...，8 通道取 ch0 降为 8 kHz
python data/prep_alimeeting.py      # 输入 Train_Ali_far/，输出 data/alimeeting/

# 2.2 AMI：segments XML → kaldi，16 kHz 降为 8 kHz
python data/prep_ami.py             # 输入 ami 音频+segments/，输出 data/ami/

# 2.3 合并 + 切分训练/验证集
python data/merge_data.py           # 输出 data/combined（训练）、data/val（验证）
```

要点：

- 验证集 = AliMeeting held-out 10 场 + AMI dev 16 场，**按会议切分**（同一会议的不同
  通道/片段不能同时出现在 train 与 val）；
- 初始中英小时比约 1.3:1（略偏中文，符合"中文优先"目标）。

### 2.4 多通道声学增强（v4b 的关键，中文 −2.68 pp）

AliMeeting 远场是 8 麦克风环形阵列：同一段对话在每个麦克风上经过**不同的房间冲激
响应**，说话人与时间标注完全不变——零标注成本的声学增强。

```bash
# 把 8 通道全部解出（209 场 × 8 = 1672 个 8 kHz wav）
bash -c 'tar -xzf Train_Ali_far.tar.gz; for each meeting: 拆 8 通道降为 8k mono'

# 生成多通道 kaldi 目录（取间隔通道 c0/c2/c4/c6，相邻麦克风高度相关）
python data/prep_ali_mc.py --channels 0,2,4,6    # 输出 data/ali_mc/
```

**域平衡（v4b→v4d 的教训）**：中文拿到 4 种真实 RIR，而英文只是同一份数据复制——
复制维持小时比 ≠ 维持有效多样性比。v4b 保持 2:1 小时比时英文退化 4.3 pp；把 AMI 复制
8 份使小时比 1:1 后英文恢复（26.89%）但中文让出 1.6 pp。**两者存在 Pareto 权衡**，
按目标语种优先度在 1:1~2:1 之间选。

---

## 3. 训练

### 3.1 配置（`conf/train_zh_en.yaml`，round2 配方）

```
init_ckpt:     上游 simu checkpoint（8 说话人模型做 4 说话人 finetune）
max_speakers:  4
chunk_size:    2000 帧（200 秒）   # 比上游 AMI 配方长 2 倍，稳定 attractor 追踪
lr:            1e-5（Adam）
batch_size:    8
max_epochs:    20
feat:          logmel23_cummn（8 kHz / 23 mel / 累积均值归一化）
```

### 3.2 运行

```bash
cp train/train_dia_fintun_real_patched.py FS-EEND/LS-EEND/train_dia_fintun_real.py
cd FS-EEND/LS-EEND
CUDA_VISIBLE_DEVICES=0 python train_dia_fintun_real.py \
  --configs conf/train_zh_en.yaml --gpus 1
```

单卡 RTX 4090D：~3.5 it/s × 553 步 ≈ 2.6 分钟/epoch，20 epoch 约 1 小时。
验证集 DER 从 ~0.45 降到 ~0.27（训练日志 `val/obj_metric`）。

### 3.3 平均最后 10 个 epoch

```bash
python train/avg_checkpoints.py \
  FS-EEND/LS-EEND/logs/spk_onl_zh_en_combined/version_0 \
  zh_en_last10.ckpt 10
```

---

## 4. 上游代码必须的 5 处修补（已包含在 patched 训练脚本中）

| # | 问题 | 修补 |
|---|---|---|
| 1 | 发布的 ckpt 键名 `dec.attractor_decoder.layers.*` 与当前代码 `dec.layers.*` 不一致，`load_state_dict` 报 strict 错误 | 加载前重命名键 |
| 2 | checkpoint 平均代码把整个 Lightning ckpt（含 `epoch` 等字符串键）做除法 → TypeError | 只取 `state_dict` |
| 3 | `AdvancedProfiler`（cProfile 级插桩）把 0.44 s/step 拖成 47 s/step | 移除 |
| 4 | **PL 1.8.6 把 `--gpus "0"`（字符串）解析成 CPU**（整数 1 才是 GPU）——训练在 CPU 上白跑数小时的元凶 | 显式 `accelerator="gpu", devices=[0]` |
| 5 | `cudnn.deterministic=True` 强制慢算法 | 改 `benchmark=True` |

另外：训练脚本末尾的 `trainer.test` 必须跳过——它用 `val_chunk=100000` 帧非流式推理，
10 万帧使 `_native_multi_head_attention` 超过 CUDA grid 上限（65535）而崩。评测统一走
`infer/streaming_infer_dia.py`（流式分块）。

**已知坑**：`data.shuffle` 配置项若为 True 会**打乱时间轴**（对流式模型毁灭性），保持
False；`consis_weight` 是死代码，我们实测提高到 5 反而使 conf 恶化（10.15%→11.14%），
不建议动。

---

## 5. 推理

```bash
cd FS-EEND/LS-EEND
python infer/streaming_infer_dia.py \
  --wav_path /path/to/16k_or_8k.wav \
  --configs conf/spk_onl_conformer_retention_enc_dec_nonautoreg_ami_infer.yaml \
  --test_from_file zh_en_last10.ckpt \
  --output_rttm out.rttm
```

- 输入 16 kHz 会由前端自动带限降采样到 8 kHz；`max_speakers=4` 由 checkpoint 固定
- 脚本同时落盘 `out.prob.npy`（逐帧 sigmoid 概率），供离线扫参（见下）
- **判决参数**：threshold 0.55 + median 11 是在 held-out 集上扫出来的最优值
  （中英一致，见 `eval/sweep_thresh.py`）

### 可选后处理：CAM++ 混合重标（`infer/hybrid_relabel.py`）

误差分解显示 LS-EEND 的 miss 全场最低（重叠检出原生），短板是身份分配（conf）。
用 CAM++（3D-Speaker 的声纹模型）对输出做「按段投票」重标：

| | 中文 conf | 中文 DER | 英文 DER |
|---|---|---|---|
| 纯 LS-EEND（round2） | 10.15% | 23.82% | 29.46% |
| + CAM++ 重标 | **5.93%** | **19.22%** | **25.96%** |

代价：引入 CAM++ 依赖（失去纯端侧单模型形态）。按部署约束选择。

---

## 6. 评测

```bash
# 指定参数打分（逐场 + 加权 + miss/FA/conf 分解）
python eval/score_at.py --prob_dir <dir_of_prob.npy> --ref_dir <dir_of_rttm> \
  --collar 0.25 --threshold 0.55 --median 11

# threshold × median 网格扫描（在 held-out 调参集上做，勿在评测集上调参）
python eval/sweep_thresh.py --prob_dir tune/zh --ref_dir tune/ref --collar 0.25
```

评分器是自研帧级实现（10 ms 网格 + 匈牙利最优映射），与 pyannote 校验一致（≤0.13 pp），
速度约快千倍。DER 口径：collar 为总窗口宽度（±collar/2），NIST/md-eval 惯例。

**评测协议**（详见主仓库对比报告）：

| 数据集 | 参考标注 | collar | 中值滤波 |
|---|---|---|---|
| AliMeeting eval 远场 4 场 | 官方 TextGrid 重建 | ±0.125 s 与 ±0.25 s 都报 | 11 |
| AMI dev | 官方 NXT words 级 | no collar | 无（上游 AMI 协议） |
| VoxConverse test 232 场 | 官方 v0.3 | ±0.125 s | 11 |

---

## 7. 复现结果与消融记录

- 完整指标对比：[RESULTS.md](RESULTS.md)
- 逐轮实验与失效分析：[experiments.md](experiments.md)
  （含 v3/v4a/v5a 等失败实验的归因，与数据截断等 3 个评测 bug 的发现过程）

## 8. 已知限制

- **`max_speakers` 训练时固定、推理不可扩展**。说话人数未知/可能 >4 的场景（如
  VoxConverse，63% 场次 >4 人）DER 随人数单调爆炸，这不是调参能解决的——需以更大
  槽位数重训（上游 simu 的 8 槽位模型在真实录音上身份追踪本就失效，不可直接替代）。
- 前端 8 kHz / 23 mel 是结构遗产；实测把 mel 提到 40 需重建两个输入层（仅 0.71% 参数），
  但从已有 checkpoint 续训的代价大于收益（val 0.27→0.50），不推荐。
