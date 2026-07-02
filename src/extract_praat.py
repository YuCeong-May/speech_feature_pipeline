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

    # Pitch/F0 in Hz, kept alongside openSMILE semitone F0 for clearer clinical reporting.
    try:
        pitch = snd.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        vals = pitch.selected_array['frequency']
        vals = vals[np.isfinite(vals) & (vals > 0)]
        out['praat_F0_mean_hz'] = _safe_float(np.mean(vals)) if len(vals) else np.nan
        out['praat_F0_std_hz'] = _safe_float(np.std(vals)) if len(vals) else np.nan
        out['praat_F0_min_hz'] = _safe_float(np.min(vals)) if len(vals) else np.nan
        out['praat_F0_max_hz'] = _safe_float(np.max(vals)) if len(vals) else np.nan
        out['praat_F0_range_hz'] = _safe_float(np.max(vals) - np.min(vals)) if len(vals) else np.nan
    except Exception:
        out['praat_F0_mean_hz'] = np.nan
        out['praat_F0_std_hz'] = np.nan
        out['praat_F0_min_hz'] = np.nan
        out['praat_F0_max_hz'] = np.nan
        out['praat_F0_range_hz'] = np.nan

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
        formant_time_step = float(cfg.get('praat_formant_time_step', 0.02))
        formant = call(
            snd, 'To Formant (burg)',
            formant_time_step,
            int(cfg.get('praat_formant_number', 5)),
            float(cfg.get('praat_formant_max_hz', 5500)),
            float(cfg.get('praat_formant_window_length', 0.05)),
            50,
        )
        duration = snd.get_total_duration()
        start_time = min(max(formant_time_step / 2, 0.0), duration)
        times = np.arange(start_time, duration, formant_time_step)
        if times.size == 0 and duration > 0:
            times = np.asarray([duration / 2], dtype=float)
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

    return out
