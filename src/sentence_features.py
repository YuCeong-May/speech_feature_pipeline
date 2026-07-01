from __future__ import annotations

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.extract_opensmile import extract_opensmile_features
from src.extract_praat import extract_praat_features
from src.extract_spectral import extract_spectral_features

FeatureExtractor = Callable[[Path, dict], dict]


def _prefixed_extract(prefix: str, extractor: FeatureExtractor, wav_path: Path, cfg: dict) -> dict:
    try:
        result = {}
        for key, value in extractor(wav_path, cfg).items():
            clean_key = key
            for source_prefix in ('praat_', 'opensmile_'):
                if clean_key.startswith(source_prefix):
                    clean_key = clean_key[len(source_prefix):]
                    break
            result[f'{prefix}_{clean_key}'] = value
        return result
    except Exception as exc:
        return {f'{prefix}_error': str(exc)}


def _write_sentence_wav(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> bool:
    start_sample = max(0, int(round(start_sec * sr)))
    end_sample = min(len(y), int(round(end_sec * sr)))
    if end_sample <= start_sample:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y[start_sample:end_sample], sr)
    return True


def extract_sentence_acoustic_features(
    sentence_metrics_csv: Path,
    source_wav_path: Path,
    cfg: dict,
    work_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Extract acoustic features directly for each sentence time window."""
    sentence_df = pd.read_csv(sentence_metrics_csv)
    sr = int(cfg.get('sample_rate', 16000))
    y, sr = librosa.load(str(source_wav_path), sr=sr, mono=True)

    rows: list[dict] = []
    for _, sentence in sentence_df.iterrows():
        sentence_id = int(sentence.get('sentence_id', len(rows) + 1))
        start = float(sentence.get('start_time', 0.0))
        end = float(sentence.get('end_time', 0.0))
        row = sentence.to_dict()
        segment_path = work_dir / f'sentence_{sentence_id:04d}.wav'
        row['sentence_wav_path'] = str(segment_path)

        if _write_sentence_wav(y, sr, start, end, segment_path):
            row.update(_prefixed_extract('sentence_spectral', extract_spectral_features, segment_path, cfg))
            row.update(_prefixed_extract('sentence_opensmile', extract_opensmile_features, segment_path, cfg))
            row.update(_prefixed_extract('sentence_praat', extract_praat_features, segment_path, cfg))
        else:
            row['sentence_acoustic_status'] = 'empty_time_window'
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return out_df
