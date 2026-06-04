from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def plot_spectrogram(wav_path: Path, output_path: Path, cfg: dict) -> Path:
    """Save a log-mel spectrogram image for quick acoustic inspection."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display

    sr = int(cfg.get('sample_rate', 16000))
    frame_length = int(cfg.get('frame_length', 2048))
    hop_length = int(cfg.get('hop_length', 512))
    n_mels = int(cfg.get('spectrogram_n_mels', 128))

    y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max) if mel.size else mel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
    )
    # Keep the title ASCII-only to avoid Matplotlib CJK glyph warnings on
    # servers that do not have Chinese fonts installed. The output filename
    # still preserves the original audio stem.
    ax.set(title='Log-Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.savefig(output_path, dpi=int(cfg.get('spectrogram_dpi', 150)))
    plt.close(fig)
    return output_path
