from __future__ import annotations

from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call


def _safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def extract_praat_features(wav_path: Path, cfg: dict) -> dict:
    snd = parselmouth.Sound(str(wav_path))
    pitch_floor = float(cfg.get('praat_pitch_floor', 75))
    pitch_ceiling = float(cfg.get('praat_pitch_ceiling', 600))

    out: dict[str, float] = {}

    # Intensity is placed here as Praat's definition is common in phonetics.
    try:
        intensity = snd.to_intensity(minimum_pitch=pitch_floor)
        vals = intensity.values[0]
        vals = vals[np.isfinite(vals)]
        out['praat_intensity_mean_db'] = _safe_float(np.mean(vals)) if len(vals) else np.nan
        out['praat_intensity_std_db'] = _safe_float(np.std(vals)) if len(vals) else np.nan
        out['praat_intensity_min_db'] = _safe_float(np.min(vals)) if len(vals) else np.nan
        out['praat_intensity_max_db'] = _safe_float(np.max(vals)) if len(vals) else np.nan
    except Exception:
        out['praat_intensity_mean_db'] = np.nan
        out['praat_intensity_std_db'] = np.nan
        out['praat_intensity_min_db'] = np.nan
        out['praat_intensity_max_db'] = np.nan

    # Formants F1-F3.
    try:
        formant = call(
            snd, 'To Formant (burg)',
            0.0,
            int(cfg.get('praat_formant_number', 5)),
            float(cfg.get('praat_formant_max_hz', 5500)),
            float(cfg.get('praat_formant_window_length', 0.025)),
            50,
        )
        duration = snd.get_total_duration()
        times = np.linspace(0.01, max(0.01, duration - 0.01), num=max(10, int(duration / 0.01)))
        for idx in [1, 2, 3]:
            vals = []
            for t in times:
                v = call(formant, 'Get value at time', idx, float(t), 'Hertz', 'Linear')
                if v and np.isfinite(v):
                    vals.append(float(v))
            vals = np.asarray(vals, dtype=float)
            out[f'praat_F{idx}_mean_hz'] = _safe_float(np.mean(vals)) if len(vals) else np.nan
            out[f'praat_F{idx}_std_hz'] = _safe_float(np.std(vals)) if len(vals) else np.nan
            out[f'praat_F{idx}_median_hz'] = _safe_float(np.median(vals)) if len(vals) else np.nan
    except Exception:
        for idx in [1, 2, 3]:
            out[f'praat_F{idx}_mean_hz'] = np.nan
            out[f'praat_F{idx}_std_hz'] = np.nan
            out[f'praat_F{idx}_median_hz'] = np.nan

    # HNR.
    try:
        harmonicity = call(snd, 'To Harmonicity (cc)', 0.01, pitch_floor, 0.1, 1.0)
        out['praat_HNR_mean_db'] = _safe_float(call(harmonicity, 'Get mean', 0, 0))
        out['praat_HNR_std_db'] = _safe_float(call(harmonicity, 'Get standard deviation', 0, 0))
    except Exception:
        out['praat_HNR_mean_db'] = np.nan
        out['praat_HNR_std_db'] = np.nan

    # Jitter / shimmer. These may fail for very noisy or unvoiced samples.
    try:
        point_process = call(snd, 'To PointProcess (periodic, cc)', pitch_floor, pitch_ceiling)
        out['praat_jitter_local'] = _safe_float(call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3))
        out['praat_jitter_rap'] = _safe_float(call(point_process, 'Get jitter (rap)', 0, 0, 0.0001, 0.02, 1.3))
        out['praat_jitter_ppq5'] = _safe_float(call(point_process, 'Get jitter (ppq5)', 0, 0, 0.0001, 0.02, 1.3))
        out['praat_shimmer_local'] = _safe_float(call([snd, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
        out['praat_shimmer_apq3'] = _safe_float(call([snd, point_process], 'Get shimmer (apq3)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
        out['praat_shimmer_apq5'] = _safe_float(call([snd, point_process], 'Get shimmer (apq5)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
        out['praat_shimmer_apq11'] = _safe_float(call([snd, point_process], 'Get shimmer (apq11)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        for k in ['jitter_local', 'jitter_rap', 'jitter_ppq5', 'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11']:
            out[f'praat_{k}'] = np.nan

    return out
