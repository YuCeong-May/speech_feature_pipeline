from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FRAME_METADATA_COLUMNS = {
    'file_id',
    'original_path',
    'wav_path',
    'frame_index',
    'frame_start_sec',
    'frame_center_sec',
    'frame_end_sec',
}


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in FRAME_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def _aggregate_window(frame_df: pd.DataFrame, start: float, end: float, prefix: str) -> dict[str, float]:
    feature_cols = _numeric_feature_columns(frame_df)
    if 'frame_center_sec' not in frame_df.columns:
        return {f'{prefix}_{col}_{stat}': np.nan for col in feature_cols for stat in ('mean', 'std', 'min', 'max')}

    mask = (frame_df['frame_center_sec'] >= start) & (frame_df['frame_center_sec'] <= end)
    window = frame_df.loc[mask, feature_cols]

    out: dict[str, float] = {}
    for col in feature_cols:
        output_prefix = col if col.startswith(f'{prefix}_') else f'{prefix}_{col}'
        vals = pd.to_numeric(window[col], errors='coerce').dropna().to_numpy(dtype=float)
        out[f'{output_prefix}_mean'] = float(np.mean(vals)) if vals.size else np.nan
        out[f'{output_prefix}_std'] = float(np.std(vals)) if vals.size else np.nan
        out[f'{output_prefix}_min'] = float(np.min(vals)) if vals.size else np.nan
        out[f'{output_prefix}_max'] = float(np.max(vals)) if vals.size else np.nan
    return out


def aggregate_sentence_acoustic_features(
    sentence_metrics_csv: Path,
    spectral_frame_csv: Path,
    praat_frame_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Aggregate default frame-level acoustic features into sentence windows."""
    sentence_df = pd.read_csv(sentence_metrics_csv)
    spectral_df = pd.read_csv(spectral_frame_csv) if spectral_frame_csv.exists() else pd.DataFrame()
    praat_df = pd.read_csv(praat_frame_csv) if praat_frame_csv.exists() else pd.DataFrame()

    rows: list[dict] = []
    for _, sentence in sentence_df.iterrows():
        start = float(sentence.get('start_time', 0.0))
        end = float(sentence.get('end_time', 0.0))
        row = sentence.to_dict()
        if not spectral_df.empty:
            row.update(_aggregate_window(spectral_df, start, end, 'spectral'))
        if not praat_df.empty:
            row.update(_aggregate_window(praat_df, start, end, 'praat'))
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return out_df
