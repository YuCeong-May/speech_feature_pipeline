# Speech Feature Pipeline

本项目用于语音心理/抑郁相关语音分析。整体工作流分为两部分：

1. 直接从音频提取声学特征：FFmpeg + openSMILE + praat-parselmouth + Librosa/SciPy。
2. 在已有标准答案转录文本时，使用 Qwen3-ForcedAligner 做强制对齐，再根据对齐时间戳计算语速、停顿、发音时间、平均音节时长等对齐后韵律指标。

本项目当前不包含 ASR 自动转录步骤。`.txt` 转录文本由外部提供，`qwen3_forced_align.py` 只负责把已有文本与音频对齐并生成时间戳。

## 1. 环境配置

建议使用两个 conda 环境：一个用于传统声学特征提取，一个用于 Qwen3-ForcedAligner。这样可以避免 `qwen-asr`、`torch`、`transformers` 与传统特征库之间的依赖冲突。

### 1.1 声学特征环境

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

### 1.2 Qwen3-ForcedAligner 环境

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

## 2. 模型下载

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

## 3. 脚本使用

### 3.1 一键提取传统声学特征

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

该脚本会先用 FFmpeg 转成标准 wav，再按“每个特征只由一个工具负责”的原则提取特征：

| 特征类别 | 具体特征 | 最终负责工具 | 主要输出前缀 |
|---|---|---|---|
| 预处理 | 转 wav、单声道、重采样 | FFmpeg | `output/work_wav/` |
| 韵律声学 | F0 / pitch | openSMILE | `opensmile_F0...` |
| 韵律声学 | loudness / volume | openSMILE | `opensmile_loudness...`、`opensmile_equivalentSoundLevel...` |
| 语音学 | intensity | praat-parselmouth | `praat_intensity...` |
| 共振峰 | F1、F2、F3 | praat-parselmouth | `praat_F1...`、`praat_F2...`、`praat_F3...` |
| 声音质量 | HNR | praat-parselmouth | `praat_HNR...` |
| 声音质量 | jitter | praat-parselmouth | `praat_jitter...` |
| 声音质量 | shimmer | praat-parselmouth | `praat_shimmer...` |
| 频谱/能量 | RMS | Librosa | `librosa_rms...` |
| 频谱 | MFCC | Librosa | `librosa_mfcc...` |
| 频谱 | PSD、bandpower | SciPy | `scipy_psd...`、`scipy_bandpower...` |
| 频谱 | LPCC | Python 自定义 LPC -> LPCC | `lpcc...` |
| 对齐后韵律 | 语速、停顿时长、停顿次数、发音时间、平均音节时长、停顿占比 | Qwen3-ForcedAligner + `calc_forced_align_metrics.py` | `output/metrics/` |

说明：openSMILE 的 eGeMAPS 本身会包含 HNR、jitter、shimmer、部分频谱等字段，但本项目只导出最终由 openSMILE 负责的 F0、loudness、volume 相关字段，避免与 praat-parselmouth、Librosa/SciPy 重复。代码中具体使用的是 `praat-parselmouth` Python 包，即 `parselmouth.Sound` 和 `parselmouth.praat.call(...)`，不是额外调用 Praat GUI 或独立 Praat 命令行程序。

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
```

### 3.2 Qwen3-ForcedAligner 强制对齐

强制对齐需要一条音频和对应的标准答案转录文本。本步骤不是 ASR，不会自动生成转录内容，只会给已有文本打时间戳：

```text
input_audio/xxx.wav
input_audio/xxx.txt
```

运行：

```bash
cd speech_feature_pipeline
conda activate qwen3-forced-aligner

python qwen3_forced_align.py \
  --audio input_audio/20230829逄JJ积极情绪事件访谈.wav \
  --text input_audio/20230829逄JJ积极情绪事件访谈.txt \
  --model ../pre_trained_models/Qwen3-ForcedAligner-0.6B \
  --language Chinese \
  --device-map cuda:0 \
  --output-json output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.json \
  --output-tsv output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.tsv
```

默认输出：

```text
output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.json
output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.tsv
```

如果需要指定输出路径：

```bash
python qwen3_forced_align.py \
  --audio input_audio/sample.wav \
  --text input_audio/sample.txt \
  --output-json output/align/sample.align.json \
  --output-tsv output/align/sample.align.tsv
```

`qwen3_forced_align.py` 参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--audio` | 必填 | 输入音频路径，支持 `wav/mp3/flac` 等常见格式。 |
| `--text` | 必填 | 与音频对应的标准答案转录文本路径。脚本会读取非空行并拼接后送入 aligner；不会做 ASR 自动转录。 |
| `--model` | `../pre_trained_models/Qwen3-ForcedAligner-0.6B` | 本地 Qwen3-ForcedAligner 模型目录。 |
| `--language` | `Chinese` | 语言名称，例如 `Chinese`、`English`。需要与模型接口支持的语言名一致。 |
| `--device-map` | `cuda:0` | 模型加载设备。常用值为 `cuda:0`、`cuda:1`、`auto` 或 `cpu`。 |
| `--dtype` | `bfloat16` | 模型权重精度，可选 `bfloat16`、`float16`、`float32`。GPU 推理建议保持 `bfloat16`。 |
| `--output-json` | 与音频同目录，后缀 `.qwen3_forced_align.json` | 对齐结果 JSON 输出路径。 |
| `--output-tsv` | 与音频同目录，后缀 `.qwen3_forced_align.tsv` | 对齐结果 TSV 输出路径，便于人工查看。 |

### 3.3 根据对齐结果计算对齐后韵律指标

该步骤不计算 F0、响度、强度、共振峰、HNR、jitter、shimmer 或频谱特征，只根据强制对齐时间戳和文本计算语速、停顿、发音时间、平均音节时长等对齐后韵律指标。

运行：

```bash
cd speech_feature_pipeline
conda activate qwen3-forced-aligner

python calc_forced_align_metrics.py \
  --align-file output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.json \
  --transcript input_audio/20230829逄JJ积极情绪事件访谈.txt \
  --pause-threshold 0.2 \
  --output-dir output/metrics
```

也可以直接使用 TSV：

```bash
python calc_forced_align_metrics.py \
  --align-file output/align/20230829逄JJ积极情绪事件访谈.qwen3_forced_align.tsv \
  --transcript input_audio/20230829逄JJ积极情绪事件访谈.txt \
  --pause-threshold 0.2 \
  --output-dir output/metrics
```

默认会过滤末尾连续 0 时长 token。这个设计用于处理“转录文本末尾有文字，但音频里实际没有发音”的情况。若要保留这些 token：

```bash
python calc_forced_align_metrics.py \
  --align-file output/align/sample.qwen3_forced_align.json \
  --transcript input_audio/sample.txt \
  --keep-trailing-zero-duration \
  --output-dir output/metrics
```

如果只需要全局 summary，不需要句子版本，可以不传 `--transcript`：

```bash
python calc_forced_align_metrics.py \
  --align-file output/align/sample.qwen3_forced_align.json \
  --output-dir output/metrics
```

`calc_forced_align_metrics.py` 参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--align-file` / `--align-json` | 必填 | Qwen3-ForcedAligner 输出文件，支持 JSON 或 TSV。`--align-json` 是兼容旧命令的别名。 |
| `--transcript` | 不填 | 原始转录文本路径。提供后会额外输出两个句子级版本；不提供时只输出全局 summary。 |
| `--pause-threshold` | `0.2` | 停顿判定阈值，单位秒。相邻 token 的间隔 `gap >= pause-threshold` 时计为一次停顿。 |
| `--keep-trailing-zero-duration` | 关闭 | 默认会丢弃末尾连续 0 时长 token；开启后保留。适合人工确认末尾 0 时长 token 仍应参与计算的情况。 |
| `--output-json` | 默认写入 `--output-dir`，后缀 `.metrics.json` | 指标 JSON 输出路径，包含全局 summary 和可选句子级结果。 |
| `--output-dir` | 对齐文件所在目录 | CSV 输出目录；如果没有显式传 `--output-json`，JSON 也会写到这里。 |

`--pause-threshold` 的含义：

```text
gap = 后一个 token 的 start_time - 前一个 token 的 end_time
```

如果 `gap >= pause-threshold`，这个间隔会被计入：

```text
pause_time_sec
pause_count
pause_ratio_percent
```

阈值越小，统计到的停顿越多；阈值越大，停顿统计越保守。当前默认 `0.2` 秒。

## 4. 输出格式与示例

### 4.1 传统声学特征输出

运行 `run.py --save_parts` 后会生成：

```text
output/features_all.csv
output/features_spectral.csv
output/features_opensmile.csv
output/features_praat.csv
output/features_all_feature_name_mapping.csv
output/work_wav/
output/logs/extract.log
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
| `praat_HNR` | praat-parselmouth 谐噪比 |
| `praat_jitter` | praat-parselmouth 频率微扰 |
| `praat_shimmer` | praat-parselmouth 振幅微扰 |
| `librosa_rms` | Librosa RMS 能量统计 |
| `librosa_mfcc` | Librosa MFCC 统计 |
| `scipy_psd` | 功率谱密度统计 |
| `scipy_bandpower` | 不同频段能量 |
| `lpcc` | Python 自定义 LPC -> LPCC 线性预测倒谱系数 |

### 4.2 强制对齐输出

`qwen3_forced_align.py` 的 JSON 输出格式：

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

### 4.3 对齐后韵律指标输出

`calc_forced_align_metrics.py` 会生成：

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

### 4.4 “呃/嗯”等语气词的两个句子版本

如果传入 `--transcript`，脚本会额外输出两个句子级 CSV：

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

## 5. 重要说明

1. 项目采用唯一责任分工：openSMILE 只导出 F0、loudness、volume；praat-parselmouth 负责 intensity、F1-F3、HNR、jitter、shimmer；Librosa/SciPy 负责 RMS、MFCC、PSD、bandpower、LPCC；Qwen3-ForcedAligner 只负责对齐后韵律指标。
2. `calc_forced_align_metrics.py` 不使用 jieba。中文汉字按单字计数，英文/数字连续串按 1 个词计。
3. 默认过滤末尾连续 0 时长 token。若音频末尾确实有发音但被对齐为 0 时长，需要人工检查后使用 `--keep-trailing-zero-duration`。
4. 对齐结果依赖转录文本质量。如果转录中包含音频里没有说出的内容，末尾或局部可能出现时间戳堆叠，需要人工记录在数据目录 README 中。
5. NAQ、QOQ 等声门特征没有纳入第一版核心流程。普通麦克风语音上直接估计可靠性有限，建议后续用专门工具单独扩展。
