# Speech Feature Pipeline

本项目用于语音心理/抑郁相关语音分析。整体工作流分为两部分：

1. 直接从音频提取声学特征：FFmpeg + openSMILE + praat-parselmouth + Librosa/SciPy。
2. 默认形成“传统声学特征提取 → Qwen3-ASR 转录 → Qwen3-ForcedAligner 强制对齐”的 pipeline；三个模块都有独立开关，也可以单独运行其中任意一个模块。

输入包括音频文件，以及可选的同名 `.txt` 转录文本；`run.py` 会先标准化音频，再按开关决定是否提取传统声学特征、是否自动转录、是否做强制对齐和句子级特征。


## 更新日志

### 2026-07-02

- 将项目整理为统一 pipeline：`run.py` 默认按“传统声学特征提取 → Qwen3-ASR 自动转录 → Qwen3-ForcedAligner 强制对齐 → 句子级特征输出”的顺序执行。
- 为传统声学特征、自动转录和强制对齐分别增加独立开关，三个模块默认开启，也可以通过参数单独关闭或独立运行。
- 新增 Qwen3-ASR 转录配置与模型下载说明，支持先自动生成同名 `.txt` 转录文本，再进入 Forced-Aligner 时间戳对齐。
- 将 Qwen3-ASR / Qwen3-ForcedAligner 与传统声学特征依赖统一到同一个 `qwen3-asr-aligner` conda 环境中，后续运行无需切换环境。
- 新增句子级声学特征输出：根据强制对齐得到的句子时间窗切分音频，并为每个句子提取传统声学特征。
- 新增对齐后的韵律统计，包括全局停顿、句内停顿、句子间间隔/停顿、语速、发音速度、平均音节时长等指标。
- 将句子间间隔定义为“当前句 `start_time` - 上一句 `end_time`”，并输出 `inter_sentence_gap_from_prev_sec`、`inter_sentence_pause_from_prev_sec`、`inter_sentence_pause_from_prev_count` 等字段。
- 调整 Praat 共振峰参数：Formants 窗长为 50 ms，窗移为 20 ms，并在配置文件中说明。
- 更新实验平台信息、Miniconda 安装脚本、模型下载命令和运行示例，便于在 Ubuntu + RTX 4090 平台上复现实验。

## 0. 实验平台

本项目当前实验平台：

```text
操作系统：Ubuntu 22.04.5 LTS
GPU：NVIDIA RTX 4090 24G （建议显存 6GB 以上）
NVIDIA Driver：575
CUDA：12.9
```

## 1. 安装 Miniconda

在运行任何流程之前，建议先安装 Miniconda，并在同一个 `qwen3-asr-aligner` 环境中同时安装传统声学特征依赖和 Qwen3-ASR / Qwen3-ForcedAligner 依赖。

如果机器上还没有 `conda`，可以运行项目提供的安装脚本：

```bash
cd speech_feature_pipeline
bash scripts/install_miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

也可以手动安装：

```bash
wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

安装完成后检查：

```bash
conda --version
```

## 2. 统一环境配置

现在只需要安装一个 conda 环境：`qwen3-asr-aligner`。先安装 Qwen3-ASR / Qwen3-ForcedAligner 所需的 Python、PyTorch 和 `qwen-asr`，再在同一个环境中补装传统声学特征依赖。后续传统声学特征提取、自动转录和强制对齐都在这个环境里运行，只通过 `run.py` 的三个模块开关控制。

> 注意：现在无需再创建或安装 `speechfeat` 环境。

### 2.1 创建 Qwen3-ASR / Qwen3-ForcedAligner 统一环境

```bash
cd speech_feature_pipeline
bash scripts/install_miniconda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

也可以手动安装：

```bash
wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

conda create -n qwen3-asr-aligner python=3.12 -y
conda activate qwen3-asr-aligner

pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0
pip install -U qwen-asr soundfile librosa pandas numpy
pip install -U "huggingface_hub[cli]" hf_transfer hf_xet
```

本项目不要求安装 `flash-attn`。如果不使用 flash attention，加载模型时不要传 `attn_implementation="flash_attention_2"`。

### 2.2 在同一环境中安装传统声学特征依赖

保持在 `qwen3-asr-aligner` 环境中继续执行：

```bash
conda activate qwen3-asr-aligner
conda install -c conda-forge ffmpeg sox libsox libsndfile -y
pip install -r requirements.txt
```

这样同一个环境同时具备：

```text
Qwen3-ASR / Qwen3-ForcedAligner
torch / torchaudio
ffmpeg
sox / libsox
libsndfile
opensmile
praat-parselmouth
librosa
scipy
numpy
pandas
PyYAML
tqdm
scikit-learn
matplotlib
```

### 2.3 检查统一环境

```bash
conda activate qwen3-asr-aligner

ffmpeg -version
sox --version
python - <<'PY'
import torch
import opensmile
import parselmouth
import librosa
import scipy
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("openSMILE:", opensmile.__version__)
print("Parselmouth:", parselmouth.__version__)
print("Librosa:", librosa.__version__)
print("SciPy:", scipy.__version__)
print("asr:", Qwen3ASRModel)
print("aligner:", Qwen3ForcedAligner)
PY
```

## 3. 模型下载

Qwen3-ASR 和 Qwen3-ForcedAligner 模型建议下载到项目同级的模型目录：

```text
../pre_trained_models/Qwen3-ASR-1.7B
../pre_trained_models/Qwen3-ASR-0.6B
../pre_trained_models/Qwen3-ForcedAligner-0.6B
```

在统一的 `qwen3-asr-aligner` 环境中使用 Hugging Face CLI 下载。默认转录模型使用 `Qwen3-ASR-1.7B`，如果显存或速度优先，可以改用 `Qwen3-ASR-0.6B`：

```bash
conda activate qwen3-asr-aligner

pip install -U "huggingface_hub[cli]" hf_transfer hf_xet
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

hf download Qwen/Qwen3-ASR-1.7B \
  --local-dir ../pre_trained_models/Qwen3-ASR-1.7B \
  --force-download

# 可选：轻量版 ASR 模型
hf download Qwen/Qwen3-ASR-0.6B \
  --local-dir ../pre_trained_models/Qwen3-ASR-0.6B \
  --force-download

hf download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir ../pre_trained_models/Qwen3-ForcedAligner-0.6B \
  --force-download
```

下载完成后检查：

```bash
ls -lh ../pre_trained_models/Qwen3-ASR-1.7B
ls -lh ../pre_trained_models/Qwen3-ForcedAligner-0.6B
```

## 4. 脚本使用

### 4.1 一键提取传统声学特征

输入音频放在：

```text
input_audio/
```

支持常见格式：`wav`、`mp3`、`m4a`、`flac`、`ogg`、`aac`、`wma`、`opus`、`webm`。

运行：

```bash
cd speech_feature_pipeline
conda activate qwen3-asr-aligner

python run.py \
  --input_dir ./input_audio \
  --output_csv ./output/features_all.csv \
  --save_parts
```

`run.py` 参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_dir` | `./input_audio` | 输入音频目录。脚本会递归读取支持的音频格式。 |
| `--output_csv` | `./output/features_all.csv` | 合并后的总特征表输出路径。 |
| `--config` | `./configs/default.yaml` | YAML 配置文件路径，用于控制采样率、openSMILE 特征集、praat-parselmouth 参数、频谱参数等。 |
| `--work_dir` | `./output/work_wav` | 标准化 wav 的临时/中间输出目录。所有输入音频会先被 FFmpeg 转到这里。 |
| `--save_parts` | 关闭 | 同时保存模块级 CSV，例如 `features_opensmile.csv`、`features_praat.csv`、`features_spectral.csv`。 |
| `--no_bilingual_row` | 关闭 | 默认 CSV 第二行会写入中英文特征名说明；开启后不写这一行，方便某些程序直接读取纯数值表。 |

新增开关：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_traditional_acoustic` | 默认开启 | 运行传统声学特征提取；该参数保留用于显式声明。 |
| `--no_traditional_acoustic` | 关闭传统声学 | 只做转录和/或强制对齐，不输出传统声学汇总 CSV。 |
| `--run_transcription` | 默认开启 | 运行 Qwen3-ASR 自动转录，先为每个音频生成 `.txt` 转录文本，再进入 Forced-Aligner；该参数保留用于显式声明。 |
| `--no_transcription` | 关闭转录 | 跳过 Qwen3-ASR，强制对齐会使用 `--transcript_dir` 或 `--input_dir` 中已有同名 `.txt`。 |
| `--transcription_output_dir` | `./output/transcripts` | Qwen3-ASR 自动转录 `.txt` 和 `.json` 元数据输出目录。 |
| `--asr_model` | `../pre_trained_models/Qwen3-ASR-1.7B` | Qwen3-ASR 本地模型目录。 |
| `--asr-language` | 自动识别 | Qwen3-ASR 识别语言；不设置时使用官方接口的自动语言识别。 |
| `--asr-max-inference-batch-size` | `32` | 传给 `Qwen3ASRModel.from_pretrained(...)` 的 batch 上限。 |
| `--asr-max-new-tokens` | `4096` | 传给 `Qwen3ASRModel.from_pretrained(...)` 的最大生成 token 数。 |
| `--run_forced_align` | 默认开启 | 执行 Forced-Aligner、对齐后韵律指标和句子级声学特征提取；该参数保留用于显式声明。 |
| `--no_forced_align` | 关闭强制对齐 | 跳过 Forced-Aligner、对齐后韵律指标和句子级声学特征。 |
| `--transcript_dir` | `--input_dir` | 转录文本目录，按音频同名 `.txt` 匹配；开启 `--run_transcription` 后会优先使用自动转录输出目录。 |
| `--sentence_output_dir` | `./output/sentence_level` | 句子级声学特征输出目录。 |


示例：

```bash
python run.py \
  --input_dir ./input_audio \
  --output_csv ./output/features_all.csv \
  --save_parts \
  --asr_model ../pre_trained_models/Qwen3-ASR-1.7B \
  --forced_align_model ../pre_trained_models/Qwen3-ForcedAligner-0.6B
```

三个模块可独立运行，例如：

```bash
# 只运行传统声学特征
python run.py --input_dir ./input_audio --output_csv ./output/features_all.csv --save_parts --no_transcription --no_forced_align

# 只运行 Qwen3-ASR 自动转录
python run.py --input_dir ./input_audio --no_traditional_acoustic --no_forced_align

# 只运行 Forced-Aligner 和对齐后句子级特征，使用已有同名 .txt
python run.py --input_dir ./input_audio --no_traditional_acoustic --no_transcription --transcript_dir ./input_audio
```

该脚本会先用 FFmpeg 转成标准 wav，再按“每个特征只由一个工具负责”的原则提取特征：

| 特征类别 | 具体特征 | 最终负责工具 | 主要输出前缀 |
|---|---|---|---|
| 预处理 | 转 wav、单声道、重采样 | FFmpeg | `output/work_wav/` |
| 韵律声学 | F0 / pitch | openSMILE | `opensmile_F0...` |
| 韵律声学 | loudness / volume | openSMILE | `opensmile_loudness...`、`opensmile_equivalentSoundLevel...` |
| 语音学 | intensity | praat-parselmouth | `praat_intensity...` |
| 共振峰 | F1、F2、F3 | praat-parselmouth | `praat_F1...`、`praat_F2...`、`praat_F3...` |
| 频谱/能量 | RMS | Librosa | `librosa_rms...` |
| 频谱 | MFCC | Librosa | `librosa_mfcc...` |
| 频谱 | PSD、bandpower | SciPy | `scipy_psd...`、`scipy_bandpower...` |
| 频谱 | LPCC | Python 自定义 LPC -> LPCC | `lpcc...` |
| 对齐后韵律 | 语速、停顿时长、停顿次数、发音时间、平均音节时长、停顿占比 | Qwen3-ForcedAligner + `run.py` 内部指标模块 | `output/metrics/` |

说明：openSMILE 负责 F0、loudness、volume 相关字段；Librosa/SciPy 负责频谱字段。Praat 相关计算通过 `praat-parselmouth` Python 包完成，即 `parselmouth.Sound` 和 `parselmouth.praat.call(...)`。

配置文件：

```text
configs/default.yaml
```

常用参数：

```yaml
sample_rate: 16000
mono: true
opensmile_feature_set: eGeMAPSv02
opensmile_feature_level: Functionals
praat_pitch_floor: 75
praat_pitch_ceiling: 600
praat_formant_max_hz: 5500
praat_formant_window_length: 0.05  # Praat Formants 窗长：50 ms
praat_formant_time_step: 0.02      # Praat Formants 窗移 / time step：20 ms
```

### 4.2 Qwen3-ASR 自动转录、ForcedAligner 强制对齐和对齐后韵律指标

主目录现在只保留 `run.py` 作为入口。自动转录、强制对齐脚本和指标脚本已经整理为内部模块：

```text
src/align/transcribe.py
src/align/forced_align.py
src/align/metrics.py
```

其中自动转录模块按 Qwen3-ASR 官方 Python 包用法加载 `Qwen3ASRModel.from_pretrained(...)`，再调用 `model.transcribe(audio=..., language=...)` 生成文本；随后本项目再把生成的 `.txt` 交给 Qwen3-ForcedAligner 做时间戳对齐。

默认情况下，`run.py` 会形成完整 pipeline；如果模型或 `qwen_asr/torch` 依赖不可用，会记录 warning 并跳过对应 Qwen3 模块。三个模块均默认开启，也都可以用 `--no_*` 参数独立关闭：

1. 标准化音频为 wav。
2. 运行传统声学汇总特征提取；可用 `--no_traditional_acoustic` 关闭。
3. 运行 Qwen3-ASR 自动转录，生成 `.txt`；可用 `--no_transcription` 关闭。
4. 运行 Qwen3-ForcedAligner 生成 JSON/TSV 时间戳；可用 `--no_forced_align` 关闭。
5. 调用内部指标模块计算语速、停顿、发音时间、平均音节时长等对齐后韵律指标。
6. 最后按句子时间窗直接提取句子级声学特征，输出句子级声学 CSV。

示例：

```bash
cd speech_feature_pipeline
conda activate qwen3-asr-aligner

python run.py \
  --input_dir ./input_audio \
  --output_csv ./output/features_all.csv \
  --save_parts \
  --run_transcription \
  --asr_model ../pre_trained_models/Qwen3-ASR-1.7B \
  --forced_align_model ../pre_trained_models/Qwen3-ForcedAligner-0.6B \
  --language Chinese \
  --device-map cuda:0 \
  --pause-threshold 0.2
```

常用 Forced-Aligner 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_traditional_acoustic` | 默认开启 | 运行传统声学特征提取；该参数保留用于显式声明。 |
| `--no_traditional_acoustic` | 关闭传统声学 | 只做转录和/或强制对齐。 |
| `--run_transcription` | 默认开启 | 开启 Qwen3-ASR 自动转录；开启后先转录，再用转录文本做 Forced-Aligner。 |
| `--no_transcription` | 关闭转录 | 跳过 Qwen3-ASR，强制对齐使用已有同名 `.txt`。 |
| `--transcription_output_dir` | `./output/transcripts` | 自动转录输出目录。 |
| `--asr_model` | `../pre_trained_models/Qwen3-ASR-1.7B` | Qwen3-ASR 本地模型目录，可改为 `Qwen3-ASR-0.6B`。 |
| `--asr-language` | 自动识别 | Qwen3-ASR 识别语言；不设置时自动识别。 |
| `--asr-max-inference-batch-size` | `32` | Qwen3-ASR 推理 batch 上限。 |
| `--asr-max-new-tokens` | `4096` | Qwen3-ASR 最大生成 token 数。 |
| `--run_forced_align` | 默认开启 | 继续运行强制对齐、对齐后韵律指标和句子级声学特征提取；该参数保留用于显式声明。 |
| `--no_forced_align` | 关闭强制对齐 | 关闭 Forced-Aligner、对齐后韵律指标和句子级特征计算。 |
| `--transcript_dir` | `--input_dir` | 手工转录文本目录；开启 `--run_transcription` 后优先使用自动转录输出目录。 |
| `--sentence_output_dir` | `./output/sentence_level` | 句子级声学特征输出目录。 |
| `--align_output_dir` | `./output/align` | 对齐 JSON/TSV 输出目录。 |
| `--metrics_output_dir` | `./output/metrics` | 对齐后韵律指标输出目录。 |
| `--forced_align_model` | `../pre_trained_models/Qwen3-ForcedAligner-0.6B` | 本地 Qwen3-ForcedAligner 模型目录。 |
| `--language` | `Chinese` | 语言名称，例如 `Chinese`、`English`。 |
| `--device-map` | `cuda:0` | 模型加载设备，例如 `cuda:0`、`auto` 或 `cpu`。 |
| `--dtype` | `bfloat16` | 模型权重精度，可选 `bfloat16`、`float16`、`float32`。 |
| `--pause-threshold` | `0.2` | 停顿判定阈值，单位秒。 |
| `--keep-trailing-zero-duration` | 关闭 | 计算对齐指标时保留末尾 0 时长 token。 |

默认输出：

```text
output/transcripts/<file_id>.txt              # 默认生成；使用 --no_transcription 时不生成
output/transcripts/<file_id>.qwen3_asr.json   # 默认生成；使用 --no_transcription 时不生成
output/align/<file_id>.qwen3_forced_align.json
output/align/<file_id>.qwen3_forced_align.tsv
output/metrics/<file_id>.qwen3_forced_align.summary.metrics.csv
output/metrics/<file_id>.qwen3_forced_align.metrics.json
output/sentence_level/<file_id>.independent_filler.sentence_acoustic.csv
output/sentence_level/<file_id>.merge_filler_to_next.sentence_acoustic.csv
```

`--pause-threshold` 的含义：

```text
gap = 后一个 token 的 start_time - 前一个 token 的 end_time
```

如果 `gap >= pause-threshold`，这个间隔会被计入 `pause_time_sec`、`pause_count` 和 `pause_ratio_percent`。阈值越小，统计到的停顿越多；阈值越大，停顿统计越保守。

## 5. 输出格式与示例

### 5.1 传统声学特征输出

运行 `run.py --save_parts` 后会生成：

```text
output/features_all.csv
output/features_spectral.csv
output/features_opensmile.csv
output/features_praat.csv
output/features_all_feature_name_mapping.csv
output/spectrograms/
output/work_wav/
output/logs/extract.log

# 默认会先生成自动转录；使用 --no_transcription 时不生成：
output/transcripts/<file_id>.txt
output/transcripts/<file_id>.qwen3_asr.json

# 如果 Forced-Aligner 成功运行，还会生成句子级声学特征：
output/sentence_level/<file_id>.independent_filler.sentence_acoustic.csv
output/sentence_level/<file_id>.merge_filler_to_next.sentence_acoustic.csv
```

注意：`output/` 下的 CSV 是运行脚本后的生成产物，不是特征定义的权威来源。修改代码或切换版本后，应重新运行 `python run.py --save_parts` 生成新结果；字段口径以当前 `src/extract_*.py` 和 `src/merge_features.py` 为准。

主要字段：

| 字段前缀 | 含义 |
|---|---|
| `opensmile_F0...` | openSMILE F0 / pitch 统计 |
| `opensmile_loudness...` | openSMILE loudness 统计 |
| `opensmile_equivalentSoundLevel...` | openSMILE volume / sound level 统计 |
| `praat_intensity` | praat-parselmouth 强度统计 |
| `praat_F1/F2/F3` | praat-parselmouth 共振峰统计 |
| `librosa_rms` | Librosa RMS 能量统计 |
| `librosa_mfcc` | Librosa MFCC 统计 |
| `scipy_psd` | 功率谱密度统计 |
| `scipy_bandpower` | 不同频段能量 |
| `lpcc` | Python 自定义 LPC -> LPCC 线性预测倒谱系数 |

### 5.2 强制对齐输出

内部 forced-align 模块的 JSON 输出格式：

```json
{
  "audio": "/path/to/audio.wav",
  "text": "/path/to/transcript.txt",
  "model": "../pre_trained_models/Qwen3-ForcedAligner-0.6B",
  "language": "Chinese",
  "items": [
    {
      "text": "如",
      "start_time": 0.0,
      "end_time": 2.72
    },
    {
      "text": "果",
      "start_time": 2.72,
      "end_time": 2.88
    }
  ]
}
```

TSV 输出格式：

```text
text    start_time    end_time
如      0.000         2.720
果      2.720         2.880
```

### 5.3 对齐后韵律指标输出

内部对齐指标模块会生成：

```text
*.summary.metrics.csv
*.metrics.json
*.independent_filler.metrics.csv
*.merge_filler_to_next.metrics.csv
```

其中 `*.summary.metrics.csv` 是后续工作流最常用的全局指标表。

`*.independent_filler.metrics.csv` 和 `*.merge_filler_to_next.metrics.csv` 为句子级指标表，其中句内停顿字段为 `pause_time`、`pause_count`、`pause_ratio`；句子间停顿按“当前句 `start_time` - 上一句 `end_time`”计算，并写入 `inter_sentence_gap_from_prev_sec`、`inter_sentence_pause_from_prev_sec`、`inter_sentence_pause_from_prev_count`。第一句没有上一句，句间 gap 记为 0。`*.metrics.json` 中各句子版本的 `summary` 还会汇总 `inter_sentence_pause_time_sec`、`inter_sentence_pause_count`、`inter_sentence_pause_ratio`、`inter_sentence_gap_time_sec_no_threshold`、`inter_sentence_positive_gap_count_no_threshold`。

字段说明：

| 字段 | 含义 |
|---|---|
| `total_duration_with_pauses_sec` | 总发音时长，首个有效发音 token 到最后一个有效发音 token，含停顿 |
| `speech_time_sec` | 发音时间，所有有效 token 的持续时间求和 |
| `pause_threshold_sec` | 停顿判定阈值，默认 0.2 秒 |
| `pause_time_sec` | 停顿时间，相邻 token 间隔大于等于阈值的 gap 求和 |
| `pause_count` | 停顿次数 |
| `pause_ratio_percent` | 停顿时间 / 总发音时长 * 100 |
| `all_gap_time_sec_no_threshold` | 不设阈值时所有正 gap 总和，仅作参考 |
| `all_positive_gap_count_no_threshold` | 不设阈值时所有正 gap 个数，仅作参考 |
| `word_count` | 词语数：中文汉字逐字计数，英文/数字连续串计 1 个词 |
| `char_count` | 总字数：中文汉字 + 英文/数字字符数 |
| `cjk_char_count` | 中文汉字数 |
| `ascii_char_count` | 英文/数字字符数 |
| `syllable_count` | 音节数：中文一个汉字按一个音节，英文/数字连续串按一个单位 |
| `speech_rate_words_per_sec` | 词语数 / 总发音时长 |
| `speech_rate_chars_per_sec` | 总字数 / 总发音时长 |
| `speech_rate_words_per_min` | 词语数 / 总发音时长 * 60 |
| `speech_rate_chars_per_min` | 总字数 / 总发音时长 * 60 |
| `articulation_rate_syllables_per_sec` | 音节数 / 发音时间，不含停顿的发音速度 |
| `articulation_rate_chars_per_sec` | 总字数 / 发音时间，不含停顿的发音速度 |
| `avg_syllable_duration_sec` | 发音时间 / 音节数 |
| `first_valid_start_sec` | 第一个有效 token 起始时间 |
| `last_valid_end_sec` | 最后一个有效 token 结束时间 |
| `valid_token_count` | 有效 token 数 |
| `valid_text` | 参与计算的有效文本 |

示例 summary：

```csv
total_duration_with_pauses_sec,speech_time_sec,pause_threshold_sec,pause_time_sec,pause_count,pause_ratio_percent,word_count,char_count,cjk_char_count,ascii_char_count,syllable_count,speech_rate_words_per_sec,speech_rate_chars_per_sec,speech_rate_words_per_min,speech_rate_chars_per_min,avg_syllable_duration_sec
83.04,50.32,0.2,28.64,35,34.489403,241,245,240,5,241,2.902216,2.950385,174.132948,177.023121,0.208797
```

### 5.4 “呃/嗯”等语气词的两个句子版本

如果默认 Forced-Aligner 成功运行且找到同名转录文本，脚本会额外输出两个句子级 CSV：

1. `independent_filler`：独立一行的“呃/嗯/啊/额”等语气词单独算一句。
2. `merge_filler_to_next`：独立一行的语气词合并到下一句开头。

示例：

```text
呃
如果说到我非常开心的事情
```

`independent_filler`：

```text
句1：呃
句2：如果说到我非常开心的事情
```

`merge_filler_to_next`：

```text
句1：呃 如果说到我非常开心的事情
```

## 6. 重要说明

1. 项目采用唯一责任分工：openSMILE 负责 F0、loudness、volume；praat-parselmouth 负责 F0、intensity、F1-F3；Librosa/SciPy 负责 RMS、MFCC、PSD、bandpower、LPCC；Qwen3-ASR 负责可选自动转录；Qwen3-ForcedAligner 负责对齐后韵律指标。
2. 中文计数按汉字逐字统计，英文/数字连续串按 1 个词计。
3. 默认过滤末尾连续 0 时长 token。若音频末尾确实有发音但被对齐为 0 时长，可人工检查后使用 `--keep-trailing-zero-duration`。
4. 对齐结果依赖转录文本质量。如果转录中包含音频里没有说出的内容，末尾或局部可能出现时间戳堆叠，建议在数据目录 README 中记录。
