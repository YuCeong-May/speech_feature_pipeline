from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'}


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger('audio_feature_extractor')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def list_audio_files(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def run_command(cmd: Iterable[str]) -> None:
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")


def flatten_dict(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}' if not prefix else f'{prefix}_{k}'
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
