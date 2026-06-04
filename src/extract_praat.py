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

    return out


def extract_praat_frame_features(wav_path: Path, cfg: dict) -> list[dict[str, float]]:
    """Return frame-level pitch, intensity, and F1-F3 samples."""
    snd = parselmouth.Sound(str(wav_path))
    pitch_floor = float(cfg.get('praat_pitch_floor', 75))
    pitch_ceiling = float(cfg.get('praat_pitch_ceiling', 600))
    time_step = float(cfg.get('frame_time_step_sec', 0.01))
    duration = snd.get_total_duration()
    if duration <= 0:
        return []

    times = np.arange(time_step / 2, duration, time_step)
    if times.size == 0:
        times = np.asarray([duration / 2], dtype=float)

    try:
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    except Exception:
        pitch = None
    try:
        intensity = snd.to_intensity(minimum_pitch=pitch_floor, time_step=time_step)
    except Exception:
        intensity = None
    try:
        formant = call(
            snd, 'To Formant (burg)',
            time_step,
            int(cfg.get('praat_formant_number', 5)),
            float(cfg.get('praat_formant_max_hz', 5500)),
            float(cfg.get('praat_formant_window_length', 0.025)),
            50,
        )
    except Exception:
        formant = None

    rows: list[dict[str, float]] = []
    for idx, t in enumerate(times):
        row = {
            'frame_index': idx,
            'frame_start_sec': _safe_float(max(0.0, float(t) - time_step / 2)),
            'frame_center_sec': _safe_float(t),
            'frame_end_sec': _safe_float(min(duration, float(t) + time_step / 2)),
        }
        if pitch is not None:
            row['praat_F0_hz'] = _safe_float(call(pitch, 'Get value at time', float(t), 'Hertz', 'Linear'))
        else:
            row['praat_F0_hz'] = np.nan
        if intensity is not None:
            row['praat_intensity_db'] = _safe_float(call(intensity, 'Get value at time', float(t), 'Cubic'))
        else:
            row['praat_intensity_db'] = np.nan
        for formant_idx in [1, 2, 3]:
            if formant is not None:
                row[f'praat_F{formant_idx}_hz'] = _safe_float(
                    call(formant, 'Get value at time', formant_idx, float(t), 'Hertz', 'Linear')
                )
            else:
                row[f'praat_F{formant_idx}_hz'] = np.nan
        rows.append(row)
    return rows
