#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from tqdm import tqdm

from src.extract_opensmile import extract_opensmile_features
from src.extract_praat import extract_praat_features
from src.extract_spectral import extract_spectral_features
from src.merge_features import save_feature_table
from src.preprocess import convert_to_wav
from src.utils import list_audio_files, load_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='One-click audio feature extractor for depression-related speech analysis.'
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
        help='Merged feature CSV path.',
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
        '--no_bilingual_row',
        action='store_true',
        help='Do not write Chinese-English bilingual description row into CSV.',
    )

    return parser.parse_args()


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

    continue_on_error = bool(cfg.get('continue_on_error', True))
    add_bilingual_row = not args.no_bilingual_row

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(input_files, desc='Extracting'):
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

            spectral_feats = extract_spectral_features(wav_path, cfg)
            smile_feats = extract_opensmile_features(wav_path, cfg)
            praat_feats = extract_praat_features(wav_path, cfg)

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
        f'Saved merged features: {args.output_csv.resolve()} | shape={df.shape}'
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


if __name__ == '__main__':
    main()
