# Speech Feature Pipeline

本项目用于语音心理/抑郁相关语音分析。整体工作流分为两部分：

1. 直接从音频提取声学特征：FFmpeg + openSMILE + praat-parselmouth + Librosa/SciPy。
2. 在已有标准答案转录文本时，使用 Qwen3-ForcedAligner 做强制对齐，再根据对齐时间戳计算语速、停顿、发音时间、平均音节时长等对齐后韵律指标。

本项目当前不包含 ASR 自动转录步骤。`.txt` 转录文本由外部提供，`run.py` 默认会在条件满足时调用内部对齐模块，把已有文本与音频对齐并生成时间戳。


## 0. 实验平台

本项目当前实验平台：

```text
操作系统：Linux Ubuntu xxx
GPU：NVIDIA RTX 4090 24G
```

## 1. 安装 Miniconda

在运行任何流程之前，建议先安装 Miniconda，并用 conda 分别管理传统声学特征环境和 Qwen3-ForcedAligner 环境。

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

## 2. 环境配置

建议使用两个 conda 环境：一个用于传统声学特征提取，一个用于 Qwen3-ForcedAligner。这样可以避免 `qwen-asr`、`torch`、`transformers` 与传统特征库之间的依赖冲突。

### 2.1 声学特征环境

项目已经提供 `environment.yml`：

```bash
cd speech_feature_pipeline

conda env create -f environment.yml
conda activate speechfeat
```

如果环境已存在，可以补装依赖：

```bash
conda activate speechfeat
conda install -c conda-forge ffmpeg sox libsox libsndfile -y
pip install -r requirements.txt
```

主要依赖包括：

```text
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

检查基础工具：

```bash
ffmpeg -version
sox --version
python - <<'PY'
import opensmile
import parselmouth
import librosa
import scipy
print("openSMILE:", opensmile.__version__)
print("Parselmouth:", parselmouth.__version__)
print("Librosa:", librosa.__version__)
print("SciPy:", scipy.__version__)
PY
```

### 2.2 Qwen3-ForcedAligner 环境

用于已有转录文本的强制对齐和对齐后韵律指标计算：

```bash
conda create -n qwen3-forced-aligner python=3.12 -y
conda activate qwen3-forced-aligner

pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0
pip install -U qwen-asr soundfile librosa pandas numpy
```

本项目不要求安装 `flash-attn`。如果不使用 flash attention，加载模型时不要传 `attn_implementation="flash_attention_2"`。

检查 CUDA 和 Qwen3-ForcedAligner：

```bash
conda activate qwen3-forced-aligner

python - <<'PY'
import torch
from qwen_asr import Qwen3ForcedAligner
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("aligner:", Qwen3ForcedAligner)
PY
```

## 3. 模型下载

Qwen3-ForcedAligner 模型建议下载到项目同级的模型目录：

```text
../pre_trained_models/Qwen3-ForcedAligner-0.6B
```

1. 用 `hf-mirror.com` clone 仓库元数据。
2. 跳过 Git LFS smudge，避免 clone 阶段长时间卡住。
3. 用 `wget -c` 单独下载 LFS 权重文件，支持断点续传和重试。

命令如下：

```bash
BASE_DIR=../pre_trained_models
MIRROR=https://hf-mirror.com
REPO=Qwen/Qwen3-ForcedAligner-0.6B
NAME=Qwen3-ForcedAligner-0.6B
DEST="${BASE_DIR}/${NAME}"

mkdir -p "${DEST}"
TMP=$(mktemp -d "/tmp/hfclone-${NAME}.XXXXXX")

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${MIRROR}/${REPO}" "${TMP}/repo"
git -C "${TMP}/repo" lfs ls-files -l | sed -E 's/^[^ ]+ [*-] //' > "${TMP}/lfs-paths"
rsync -a --exclude-from="${TMP}/lfs-paths" "${TMP}/repo/" "${DEST}/"

while IFS= read -r path; do
  [ -z "${path}" ] && continue
  mkdir -p "${DEST}/$(dirname "${path}")"
  wget -c --tries=20 --timeout=30 --waitretry=5 \
    -O "${DEST}/${path}" \
    "${MIRROR}/${REPO}/resolve/main/$(printf '%s' "${path}" | sed 's/ /%20/g')"
done < "${TMP}/lfs-paths"

rm -rf "${TMP}"
```

下载完成后检查：

```bash
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
conda activate speechfeat

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
| `--save_frame_level` | 默认开启 | 帧级 CSV 输出开关，会为每个音频生成频谱帧表和 Praat 帧表；该参数保留用于显式声明。 |
| `--no_frame_level` | 关闭 | 关闭默认帧级 CSV 输出。 |
| `--frame_output_dir` | `./output/frame_level` | 帧级 CSV 输出目录。 |
| `--spectrogram_dir` | `./output/spectrograms` | 传统声学特征提取时同步输出频谱图 PNG。 |
| `--no_spectrogram` | 关闭 | 跳过频谱图绘制。 |
| `--run_forced_align` | 默认开启 | 帧级特征完成后，在模型和依赖可用时执行 Forced-Aligner、对齐后韵律指标和句子级声学聚合；该参数保留用于显式声明。 |
| `--no_forced_align` | 关闭 | 关闭默认 Forced-Aligner 和句子级特征计算。 |
| `--transcript_dir` | `--input_dir` | 转录文本目录，按音频同名 `.txt` 匹配。 |
| `--sentence_output_dir` | `./output/sentence_level` | 句子级声学特征输出目录，由帧级 CSV 按句子时间窗聚合得到。 |

性能说明：默认帧级 Praat 采样间隔为 `frame_time_step_sec: 0.05` 秒；逐帧 LPCC 默认关闭，因为长访谈上逐帧 LPCC 计算量较大。整段 LPCC 汇总仍会正常输出。如确实需要逐帧 LPCC，可在 `configs/default.yaml` 中设置 `frame_spectral_include_lpcc: true`。

示例：

```bash
python run.py \
  --input_dir ./input_audio \
  --output_csv ./output/features_all.csv \
  --save_parts \
  --transcript_dir ./input_audio \
  --forced_align_model ../pre_trained_models/Qwen3-ForcedAligner-0.6B
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

说明：openSMILE 的 eGeMAPS 本身会包含部分频谱字段，但本项目只保留 F0、loudness、volume 相关字段，避免与 Librosa/SciPy 重复。代码中具体使用的是 `praat-parselmouth` Python 包，即 `parselmouth.Sound` 和 `parselmouth.praat.call(...)`，不是额外调用 Praat GUI 或独立 Praat 命令行程序。

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
frame_time_step_sec: 0.05
frame_spectral_include_lpcc: false
```

### 4.2 Qwen3-ForcedAligner 强制对齐和对齐后韵律指标

主目录现在只保留 `run.py` 作为入口。强制对齐脚本和指标脚本已经整理为内部模块：

```text
src/align/forced_align.py
src/align/metrics.py
```

默认情况下，`run.py` 会按照下面顺序执行；如果模型或 `qwen_asr/torch` 依赖不可用，会记录 warning 并跳过 Forced-Aligner，可用 `--no_forced_align` 显式关闭：

1. 先提取传统声学汇总特征，并同步绘制频谱图。
2. 默认提取帧级频谱特征和帧级 Praat 特征。
3. 再按音频 `file_id` 到 `--transcript_dir` 中寻找同名 `.txt` 转录文本。
4. 调用 Qwen3-ForcedAligner 生成 JSON/TSV 时间戳。
5. 调用内部指标模块计算语速、停顿、发音时间、平均音节时长等对齐后韵律指标。
6. 最后按句子时间窗聚合帧级声学特征，输出句子级声学 CSV。

示例：

```bash
cd speech_feature_pipeline
conda activate qwen3-forced-aligner

python run.py \
  --input_dir ./input_audio \
  --output_csv ./output/features_all.csv \
  --save_parts \
  --transcript_dir ./input_audio \
  --forced_align_model ../pre_trained_models/Qwen3-ForcedAligner-0.6B \
  --language Chinese \
  --device-map cuda:0 \
  --pause-threshold 0.2
```

常用 Forced-Aligner 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_forced_align` | 默认开启 | 帧级特征后继续运行强制对齐、对齐后韵律指标和句子级声学聚合；该参数保留用于显式声明。 |
| `--no_forced_align` | 关闭 | 关闭默认强制对齐和句子级特征计算。 |
| `--transcript_dir` | `--input_dir` | 转录文本目录，按音频同名 `.txt` 匹配。 |
| `--sentence_output_dir` | `./output/sentence_level` | 句子级声学特征输出目录，由帧级 CSV 按句子时间窗聚合得到。 |
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

# 默认还会生成帧级输出：
output/frame_level/<file_id>.spectral_frames.csv
output/frame_level/<file_id>.praat_frames.csv

# 如果 Forced-Aligner 成功运行，还会生成句子级声学聚合：
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

1. 项目采用唯一责任分工：openSMILE 只导出 F0、loudness、volume；praat-parselmouth 负责 F0、intensity、F1-F3；Librosa/SciPy 负责 RMS、MFCC、PSD、bandpower、LPCC；Qwen3-ForcedAligner 只负责对齐后韵律指标。
2. 内部对齐指标模块不使用 jieba。中文汉字按单字计数，英文/数字连续串按 1 个词计。
3. 默认过滤末尾连续 0 时长 token。若音频末尾确实有发音但被对齐为 0 时长，需要人工检查后使用 `--keep-trailing-zero-duration`。
4. 对齐结果依赖转录文本质量。如果转录中包含音频里没有说出的内容，末尾或局部可能出现时间戳堆叠，需要人工记录在数据目录 README 中。
5. NAQ、QOQ 等声门特征没有纳入第一版核心流程。普通麦克风语音上直接估计可靠性有限，建议后续用专门工具单独扩展。
