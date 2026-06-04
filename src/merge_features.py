from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_NAME_ZH: dict[str, str] = {
    # 基础信息
    'file_id': '文件编号',
    'original_path': '原始音频路径',
    'wav_path': '标准化WAV路径',
    'status': '处理状态',
    'error': '错误信息',

    # 时长
    'duration_sec': '音频总时长_秒',

    # Praat: F0
    'praat_F0_mean_hz': 'Praat基频F0均值_Hz',
    'praat_F0_std_hz': 'Praat基频F0标准差_Hz',
    'praat_F0_min_hz': 'Praat基频F0最小值_Hz',
    'praat_F0_max_hz': 'Praat基频F0最大值_Hz',
    'praat_F0_range_hz': 'Praat基频F0范围_Hz',
}


STAT_NAME_ZH: dict[str, str] = {
    'mean': '均值',
    'std': '标准差',
    'min': '最小值',
    'max': '最大值',
    'median': '中位数',
}


FORMANT_ORDINAL_ZH: dict[str, str] = {
    '1': '一',
    '2': '二',
    '3': '三',
}


def _stat_suffix_name(stat: str) -> str:
    return STAT_NAME_ZH.get(stat, stat)


def chinese_name_for_feature(col: str) -> str:
    """
    根据英文列名生成中文解释。
    对固定列名使用人工映射；
    对批量特征，例如 librosa_mfcc1_mean、lpcc3、opensmile_*，使用规则生成。
    """
    if col in FEATURE_NAME_ZH:
        return FEATURE_NAME_ZH[col]

    lower = col.lower()

    m = re.fullmatch(r'librosa_rms_(mean|std|min|max)', lower)
    if m:
        return f'Librosa均方根能量RMS{_stat_suffix_name(m.group(1))}'

    m = re.fullmatch(r'librosa_mfcc(\d+)_(mean|std|min|max)', lower)
    if m:
        return f'Librosa梅尔频率倒谱系数MFCC{m.group(1)}{_stat_suffix_name(m.group(2))}'

    m = re.fullmatch(r'scipy_psd_(mean|std|min|max)', lower)
    if m:
        return f'SciPy功率谱密度PSD{_stat_suffix_name(m.group(1))}'

    m = re.fullmatch(r'scipy_bandpower_(\d+)_(\d+)_hz', lower)
    if m:
        return f'SciPy频带能量_{m.group(1)}到{m.group(2)}Hz'

    m = re.fullmatch(r'lpcc(\d+)', lower)
    if m:
        return f'线性预测倒谱系数LPCC{m.group(1)}'

    m = re.fullmatch(r'praat_intensity_(mean|std|min|max)_db', lower)
    if m:
        return f'Praat强度{_stat_suffix_name(m.group(1))}_dB'

    m = re.fullmatch(r'praat_f([123])_(mean|std|median)_hz', lower)
    if m:
        ordinal = FORMANT_ORDINAL_ZH[m.group(1)]
        return f'Praat第{ordinal}共振峰F{m.group(1)}{_stat_suffix_name(m.group(2))}_Hz'

    # openSMILE / eGeMAPS 常见字段。顺序要先处理更具体的字段。
    if 'equivalentsoundlevel' in lower:
        return 'openSMILE等效声级或音量相关指标'

    if 'f0' in lower or 'pitch' in lower:
        return 'openSMILE基频F0或音高相关指标'

    if 'loudness' in lower:
        return 'openSMILE响度相关指标'

    if 'intensity' in lower:
        return '强度相关指标'

    if 'energy' in lower or 'rms' in lower:
        return '能量或音量相关指标'

    if 'formant' in lower or lower.startswith('f1') or lower.startswith('f2') or lower.startswith('f3'):
        return '共振峰相关指标'

    if 'mfcc' in lower:
        return '梅尔频率倒谱系数MFCC相关指标'

    if 'alpha' in lower:
        return '频谱Alpha比率相关指标'

    if 'hammarberg' in lower:
        return 'Hammarberg频谱指数相关指标'

    if 'slope' in lower:
        return '频谱斜率相关指标'

    if 'flux' in lower:
        return '频谱通量相关指标'

    if 'spectral' in lower:
        return '频谱相关指标'

    if 'voiced' in lower or 'voicing' in lower:
        return '浊音或发声相关指标'

    if 'pause' in lower:
        return '停顿相关指标'

    if 'speech' in lower:
        return '语音或语速相关指标'

    if 'duration' in lower:
        return '时长相关指标'

    # 没有匹配到时，返回原列名，避免乱翻译
    return col


def build_bilingual_row(columns: list[str]) -> dict[str, str]:
    """
    构造 CSV 第二行：中文名 / 英文名。
    """
    return {
        col: f'{chinese_name_for_feature(col)} / {col}'
        for col in columns
    }


def save_feature_name_mapping(df: pd.DataFrame, csv_path: Path) -> None:
    """
    额外保存一个字段名映射表，方便老师查看每个指标的中英文含义。
    """
    mapping_path = csv_path.with_name(csv_path.stem + '_feature_name_mapping.csv')

    mapping = pd.DataFrame(
        {
            'english_name': list(df.columns),
            'chinese_name': [chinese_name_for_feature(col) for col in df.columns],
            'bilingual_name': [
                f'{chinese_name_for_feature(col)} / {col}'
                for col in df.columns
            ],
        }
    )

    mapping.to_csv(mapping_path, index=False, encoding='utf-8-sig')


def save_feature_table(
    rows: list[dict[str, Any]],
    output_csv: Path,
    add_bilingual_row: bool = True,
) -> pd.DataFrame:
    """
    保存特征表。

    参数
    ----
    rows:
        每个音频文件对应一行特征字典。

    output_csv:
        输出 CSV 路径。

    add_bilingual_row:
        True:
            CSV 第一行为英文列名；
            CSV 第二行为“中文名 / 英文名”；
            CSV 第三行开始为真实数据。

        False:
            CSV 第一行为英文列名；
            CSV 第二行开始为真实数据。
            这种格式更适合机器学习程序直接读取。

    返回
    ----
    df:
        不包含中文解释行的原始 DataFrame。
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    # 不管是否在主 CSV 中加入双语行，都单独保存映射表
    save_feature_name_mapping(df, output_csv)

    if add_bilingual_row and not df.empty:
        bilingual_row = build_bilingual_row(list(df.columns))
        df_to_save = pd.concat(
            [pd.DataFrame([bilingual_row]), df],
            ignore_index=True,
        )
    else:
        df_to_save = df

    df_to_save.to_csv(output_csv, index=False, encoding='utf-8-sig')

    return df
