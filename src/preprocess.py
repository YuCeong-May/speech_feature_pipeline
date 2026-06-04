from __future__ import annotations

from pathlib import Path

import librosa

from .utils import run_command


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000, mono: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
    ]
    if mono:
        cmd += ['-ac', '1']
    cmd += ['-ar', str(sample_rate), '-sample_fmt', 's16', str(output_path)]
    run_command(cmd)
    return output_path


def get_duration_seconds(wav_path: Path) -> float:
    return float(librosa.get_duration(path=str(wav_path)))
