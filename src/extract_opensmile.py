from __future__ import annotations

from pathlib import Path

import numpy as np
import opensmile


OPENSMILE_FINAL_PATTERNS = (
    'F0semitoneFrom27.5Hz',
    'loudness',
    'equivalentSoundLevel',
)


def _get_feature_set(name: str):
    if not hasattr(opensmile.FeatureSet, name):
        raise ValueError(f'Unsupported opensmile FeatureSet: {name}')
    return getattr(opensmile.FeatureSet, name)


def _get_feature_level(name: str):
    if not hasattr(opensmile.FeatureLevel, name):
        raise ValueError(f'Unsupported opensmile FeatureLevel: {name}')
    return getattr(opensmile.FeatureLevel, name)


def extract_opensmile_features(wav_path: Path, cfg: dict) -> dict:
    smile = opensmile.Smile(
        feature_set=_get_feature_set(cfg.get('opensmile_feature_set', 'eGeMAPSv02')),
        feature_level=_get_feature_level(cfg.get('opensmile_feature_level', 'Functionals')),
    )
    df = smile.process_file(str(wav_path))
    if df.empty:
        return {}
    row = df.iloc[0]
    result = {}
    for col, val in row.items():
        # Keep openSMILE's final responsibility narrow; duplicate voice-quality
        # and spectral fields are exported by Praat and Librosa/SciPy instead.
        if not any(pattern in col for pattern in OPENSMILE_FINAL_PATTERNS):
            continue
        try:
            result[f'opensmile_{col}'] = float(val) if np.isfinite(val) else np.nan
        except Exception:
            result[f'opensmile_{col}'] = val
    return result
