from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import welch


def _safe(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _stats(prefix: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan, f'{prefix}_min': np.nan, f'{prefix}_max': np.nan}
    return {
        f'{prefix}_mean': _safe(np.mean(values)),
        f'{prefix}_std': _safe(np.std(values)),
        f'{prefix}_min': _safe(np.min(values)),
        f'{prefix}_max': _safe(np.max(values)),
    }


def _lpc_coefficients(y: np.ndarray, order: int) -> np.ndarray:
    # Autocorrelation LPC. Returns a[0]=1, a[1:order+1].
    y = np.asarray(y, dtype=float)
    if len(y) <= order + 1 or np.allclose(y, 0):
        return np.full(order + 1, np.nan)
    y = y - np.mean(y)
    autocorr = np.correlate(y, y, mode='full')[len(y)-1:len(y)+order]
    if autocorr[0] <= 1e-12:
        return np.full(order + 1, np.nan)
    try:
        a_tail = solve_toeplitz((autocorr[:order], autocorr[:order]), -autocorr[1:order+1])
        return np.concatenate([[1.0], a_tail])
    except Exception:
        return np.full(order + 1, np.nan)


def _lpc_to_lpcc(a: np.ndarray, cep_order: int) -> np.ndarray:
    # LPC -> cepstral coefficients. c[0] omitted in output.
    p = len(a) - 1
    c = np.zeros(cep_order + 1, dtype=float)
    if not np.all(np.isfinite(a)):
        return np.full(cep_order, np.nan)
    for n in range(1, cep_order + 1):
        acc = 0.0
        for k in range(1, n):
            if n - k <= p:
                acc += (k / n) * c[k] * a[n - k]
        c[n] = -a[n] - acc if n <= p else -acc
    return c[1:]


def extract_spectral_features(wav_path: Path, cfg: dict) -> dict:
    sr = int(cfg.get('sample_rate', 16000))
    y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    frame_length = int(cfg.get('frame_length', 2048))
    hop_length = int(cfg.get('hop_length', 512))

    out: dict[str, float] = {'duration_sec': _safe(duration)}

    # RMS/volume summary. Pause/speech-time metrics are intentionally not
    # computed here; they are derived from Qwen3-ForcedAligner timestamps.
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    out.update(_stats('librosa_rms', rms))

    # MFCC summary, independent of openSMILE; kept with librosa prefix for transparent comparison.
    n_mfcc = int(cfg.get('mfcc_n', 13))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    for i in range(n_mfcc):
        out.update(_stats(f'librosa_mfcc{i+1}', mfcc[i]))

    # PSD with Welch.
    nperseg = min(int(cfg.get('psd_nperseg', 1024)), len(y)) if len(y) > 0 else int(cfg.get('psd_nperseg', 1024))
    if nperseg > 8:
        freqs, pxx = welch(y, fs=sr, nperseg=nperseg)
        out.update(_stats('scipy_psd', pxx))
        bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            out[f'scipy_bandpower_{lo}_{hi}_hz'] = _safe(np.trapz(pxx[mask], freqs[mask])) if np.any(mask) else np.nan
    else:
        out.update(_stats('scipy_psd', np.array([])))

    # LPCC from whole signal LPC. For clinical usage, consider also frame-level LPCC in a later version.
    lpc_order = int(cfg.get('lpc_order', 16))
    lpcc_order = int(cfg.get('lpcc_order', 13))
    a = _lpc_coefficients(y, lpc_order)
    lpcc = _lpc_to_lpcc(a, lpcc_order)
    for i, v in enumerate(lpcc, start=1):
        out[f'lpcc{i}'] = _safe(v)

    return out


def _frame_lpcc(y: np.ndarray, frame_length: int, hop_length: int, lpc_order: int, lpcc_order: int) -> np.ndarray:
    if y.size == 0:
        return np.empty((0, lpcc_order), dtype=float)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    rows = []
    for frame in frames:
        a = _lpc_coefficients(frame, lpc_order)
        rows.append(_lpc_to_lpcc(a, lpcc_order))
    return np.asarray(rows, dtype=float)


def extract_spectral_frame_features(wav_path: Path, cfg: dict) -> list[dict[str, float]]:
    """Return frame-level spectral features for optional time-series export."""
    sr = int(cfg.get('sample_rate', 16000))
    y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
    frame_length = int(cfg.get('frame_length', 2048))
    hop_length = int(cfg.get('hop_length', 512))
    n_mfcc = int(cfg.get('mfcc_n', 13))
    lpc_order = int(cfg.get('lpc_order', 16))
    lpcc_order = int(cfg.get('lpcc_order', 13))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    stft_power = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    lpcc = _frame_lpcc(y, frame_length, hop_length, lpc_order, lpcc_order)

    frame_count = max(rms.shape[0], mfcc.shape[1], stft_power.shape[1])
    centers = librosa.frames_to_time(np.arange(frame_count), sr=sr, hop_length=hop_length, n_fft=frame_length)
    half_window = frame_length / (2 * sr)
    bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]

    rows: list[dict[str, float]] = []
    for idx in range(frame_count):
        row: dict[str, float] = {
            'frame_index': idx,
            'frame_start_sec': _safe(max(0.0, centers[idx] - half_window)),
            'frame_center_sec': _safe(centers[idx]),
            'frame_end_sec': _safe(centers[idx] + half_window),
            'librosa_rms': _safe(rms[idx]) if idx < rms.shape[0] else np.nan,
        }
        for mfcc_idx in range(n_mfcc):
            row[f'librosa_mfcc{mfcc_idx + 1}'] = _safe(mfcc[mfcc_idx, idx]) if idx < mfcc.shape[1] else np.nan
        if idx < stft_power.shape[1]:
            pxx = stft_power[:, idx]
            row.update(_stats('frame_psd', pxx))
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                row[f'frame_bandpower_{lo}_{hi}_hz'] = _safe(np.trapz(pxx[mask], freqs[mask])) if np.any(mask) else np.nan
        else:
            row.update(_stats('frame_psd', np.array([])))
            for lo, hi in bands:
                row[f'frame_bandpower_{lo}_{hi}_hz'] = np.nan
        for lpcc_idx in range(lpcc_order):
            row[f'frame_lpcc{lpcc_idx + 1}'] = _safe(lpcc[idx, lpcc_idx]) if idx < lpcc.shape[0] else np.nan
        rows.append(row)
    return rows
