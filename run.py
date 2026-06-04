#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.extract_opensmile import extract_opensmile_features
from src.extract_praat import extract_praat_features, extract_praat_frame_features
from src.extract_spectral import extract_spectral_features, extract_spectral_frame_features
from src.merge_features import save_feature_table
from src.plot_spectrogram import plot_spectrogram
from src.preprocess import convert_to_wav
from src.utils import list_audio_files, load_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='One-click audio feature extractor and optional forced-alignment metric runner.'
    )

    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('./input_audio'),
        help='Folder containing input audio files.',
    )

    parser.add_argument(
        '--output_csv',
        type=Path,
        default=Path('./output/features_all.csv'),
        help='Merged traditional acoustic feature CSV path.',
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('./configs/default.yaml'),
        help='YAML config path.',
    )

    parser.add_argument(
        '--work_dir',
        type=Path,
        default=Path('./output/work_wav'),
        help='Folder for standardized wav files.',
    )

    parser.add_argument(
        '--save_parts',
        action='store_true',
        help='Also save module-level CSV files.',
    )

    parser.add_argument(
        '--save_frame_level',
        action='store_true',
        help='Switch on frame-level CSV export for spectral, pitch/intensity, and formant time series.',
    )

    parser.add_argument(
        '--frame_output_dir',
        type=Path,
        default=Path('./output/frame_level'),
        help='Folder for optional frame-level CSV files.',
    )

    parser.add_argument(
        '--spectrogram_dir',
        type=Path,
        default=Path('./output/spectrograms'),
        help='Folder for spectrogram PNG files generated during traditional acoustic extraction.',
    )

    parser.add_argument(
        '--no_spectrogram',
        action='store_true',
        help='Skip spectrogram plotting.',
    )

    parser.add_argument(
        '--no_bilingual_row',
        action='store_true',
        help='Do not write Chinese-English bilingual description row into CSV.',
    )

    parser.add_argument(
        '--run_forced_align',
        action='store_true',
        help='After traditional acoustic extraction, run Qwen3-ForcedAligner and alignment metrics for audio/text pairs.',
    )

    parser.add_argument(
        '--transcript_dir',
        type=Path,
        default=None,
        help='Folder containing transcript .txt files. Defaults to --input_dir.',
    )

    parser.add_argument(
        '--align_output_dir',
        type=Path,
        default=Path('./output/align'),
        help='Folder for Qwen3 forced-alignment JSON/TSV outputs.',
    )

    parser.add_argument(
        '--metrics_output_dir',
        type=Path,
        default=Path('./output/metrics'),
        help='Folder for forced-alignment metric outputs.',
    )

    parser.add_argument(
        '--forced_align_model',
        default='../pre_trained_models/Qwen3-ForcedAligner-0.6B',
        help='Local Qwen3-ForcedAligner model directory.',
    )

    parser.add_argument('--language', default='Chinese', help='Forced-aligner language name, e.g. Chinese or English.')
    parser.add_argument('--device-map', default='cuda:0', help='Forced-aligner device map, e.g. cuda:0, auto, or cpu.')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--pause-threshold', type=float, default=0.2, help='Pause threshold in seconds for alignment metrics.')
    parser.add_argument(
        '--keep-trailing-zero-duration',
        action='store_true',
        help='Keep trailing zero-duration alignment tokens when calculating alignment metrics.',
    )

    return parser.parse_args()


def _save_frame_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _find_transcript(transcript_dir: Path, file_id: str) -> Path | None:
    direct = transcript_dir / f'{file_id}.txt'
    if direct.exists():
        return direct
    matches = sorted(p for p in transcript_dir.rglob('*.txt') if p.stem == file_id)
    return matches[0] if matches else None


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _run_forced_alignment(args: argparse.Namespace, wav_jobs: list[tuple[str, Path]], logger) -> None:
    transcript_dir = args.transcript_dir or args.input_dir
    args.align_output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output_dir.mkdir(parents=True, exist_ok=True)

    align_script = Path(__file__).parent / 'src' / 'align' / 'forced_align.py'
    metrics_script = Path(__file__).parent / 'src' / 'align' / 'metrics.py'

    for file_id, wav_path in tqdm(wav_jobs, desc='Forced aligning'):
        transcript_path = _find_transcript(transcript_dir, file_id)
        if transcript_path is None:
            logger.warning(f'Skip forced alignment for {file_id}: transcript not found in {transcript_dir}')
            continue

        output_json = args.align_output_dir / f'{file_id}.qwen3_forced_align.json'
        output_tsv = args.align_output_dir / f'{file_id}.qwen3_forced_align.tsv'

        _run_command([
            sys.executable,
            str(align_script),
            '--audio',
            str(wav_path),
            '--text',
            str(transcript_path),
            '--model',
            args.forced_align_model,
            '--language',
            args.language,
            '--device-map',
            args.device_map,
            '--dtype',
            args.dtype,
            '--output-json',
            str(output_json),
            '--output-tsv',
            str(output_tsv),
        ])

        metrics_cmd = [
            sys.executable,
            str(metrics_script),
            '--align-file',
            str(output_json),
            '--transcript',
            str(transcript_path),
            '--pause-threshold',
            str(args.pause_threshold),
            '--output-dir',
            str(args.metrics_output_dir),
        ]
        if args.keep_trailing_zero_duration:
            metrics_cmd.append('--keep-trailing-zero-duration')
        _run_command(metrics_cmd)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    log_path = args.output_csv.parent / 'logs' / 'extract.log'
    logger = setup_logger(log_path)

    input_files = list_audio_files(args.input_dir)
    if not input_files:
        logger.warning(f'No audio files found in {args.input_dir.resolve()}')
        return

    logger.info(f'Found {len(input_files)} audio files.')

    rows_all: list[dict] = []
    rows_smile: list[dict] = []
    rows_praat: list[dict] = []
    rows_spectral: list[dict] = []
    wav_jobs: list[tuple[str, Path]] = []

    continue_on_error = bool(cfg.get('continue_on_error', True))
    add_bilingual_row = not args.no_bilingual_row
    save_spectrogram = not args.no_spectrogram

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(input_files, desc='Extracting traditional acoustic features'):
        file_id = audio_path.stem
        wav_path = args.work_dir / f'{file_id}.wav'

        base = {
            'file_id': file_id,
            'original_path': str(audio_path),
            'wav_path': str(wav_path),
        }

        try:
            convert_to_wav(
                audio_path,
                wav_path,
                sample_rate=int(cfg.get('sample_rate', 16000)),
                mono=bool(cfg.get('mono', True)),
            )
            wav_jobs.append((file_id, wav_path))

            spectral_feats = extract_spectral_features(wav_path, cfg)
            smile_feats = extract_opensmile_features(wav_path, cfg)
            praat_feats = extract_praat_features(wav_path, cfg)

            if save_spectrogram:
                spectrogram_path = args.spectrogram_dir / f'{file_id}.spectrogram.png'
                if importlib.util.find_spec('matplotlib') is None:
                    logger.warning(
                        f'Skip spectrogram for {file_id}: matplotlib is not installed. '
                        'Install matplotlib to enable spectrogram plotting.'
                    )
                else:
                    plot_spectrogram(wav_path, spectrogram_path, cfg)

            if args.save_frame_level:
                spectral_frame_rows = [
                    {**base, **row}
                    for row in extract_spectral_frame_features(wav_path, cfg)
                ]
                praat_frame_rows = [
                    {**base, **row}
                    for row in extract_praat_frame_features(wav_path, cfg)
                ]
                _save_frame_csv(spectral_frame_rows, args.frame_output_dir / f'{file_id}.spectral_frames.csv')
                _save_frame_csv(praat_frame_rows, args.frame_output_dir / f'{file_id}.praat_frames.csv')

            row_all = {
                **base,
                'status': 'ok',
                **spectral_feats,
                **smile_feats,
                **praat_feats,
            }

            rows_all.append(row_all)
            rows_spectral.append({**base, 'status': 'ok', **spectral_feats})
            rows_smile.append({**base, 'status': 'ok', **smile_feats})
            rows_praat.append({**base, 'status': 'ok', **praat_feats})

        except Exception as e:
            logger.error(f'Failed: {audio_path}\n{traceback.format_exc()}')

            err_row = {
                **base,
                'status': 'failed',
                'error': str(e),
            }

            rows_all.append(err_row)
            rows_spectral.append(err_row)
            rows_smile.append(err_row)
            rows_praat.append(err_row)

            if not continue_on_error:
                raise

    df = save_feature_table(
        rows_all,
        args.output_csv,
        add_bilingual_row=add_bilingual_row,
    )

    logger.info(
        f'Saved merged traditional acoustic features: {args.output_csv.resolve()} | shape={df.shape}'
    )

    if args.save_parts:
        out_dir = args.output_csv.parent

        save_feature_table(
            rows_spectral,
            out_dir / 'features_spectral.csv',
            add_bilingual_row=add_bilingual_row,
        )

        save_feature_table(
            rows_smile,
            out_dir / 'features_opensmile.csv',
            add_bilingual_row=add_bilingual_row,
        )

        save_feature_table(
            rows_praat,
            out_dir / 'features_praat.csv',
            add_bilingual_row=add_bilingual_row,
        )

        logger.info(f'Saved part-level CSV files under: {out_dir.resolve()}')

    if args.run_forced_align:
        logger.info('Traditional acoustic extraction finished. Starting Qwen3-ForcedAligner and alignment metrics.')
        _run_forced_alignment(args, wav_jobs, logger)


if __name__ == '__main__':
    main()
