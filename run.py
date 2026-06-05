#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from src.extract_opensmile import extract_opensmile_features
from src.extract_praat import extract_praat_features
from src.extract_spectral import extract_spectral_features
from src.merge_features import save_feature_table
from src.preprocess import convert_to_wav
from src.sentence_features import extract_sentence_acoustic_features
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
        '--no_bilingual_row',
        action='store_true',
        help='Do not write Chinese-English bilingual description row into CSV.',
    )

    parser.add_argument(
        '--run_transcription',
        action='store_true',
        default=False,
        help='Run Qwen3-ASR transcription before forced alignment. Disabled by default.',
    )

    parser.add_argument(
        '--transcription_output_dir',
        type=Path,
        default=Path('./output/transcripts'),
        help='Folder for Qwen3-ASR transcript .txt and metadata .json outputs.',
    )

    parser.add_argument(
        '--asr_model',
        default='../pre_trained_models/Qwen3-ASR-1.7B',
        help='Local Qwen3-ASR model directory used when --run_transcription is enabled.',
    )

    parser.add_argument(
        '--asr-language',
        dest='asr_language',
        default=None,
        help='Qwen3-ASR language name. Leave unset for automatic language detection.',
    )

    parser.add_argument(
        '--asr-max-inference-batch-size',
        type=int,
        default=32,
        help='Qwen3-ASR max_inference_batch_size passed to Qwen3ASRModel.from_pretrained.',
    )

    parser.add_argument(
        '--asr-max-new-tokens',
        type=int,
        default=4096,
        help='Qwen3-ASR max_new_tokens passed to Qwen3ASRModel.from_pretrained.',
    )

    parser.add_argument(
        '--run_forced_align',
        action='store_true',
        default=True,
        help='Run Qwen3-ForcedAligner and sentence-level metrics after acoustic extraction. Enabled by default.',
    )

    parser.add_argument(
        '--no_forced_align',
        dest='run_forced_align',
        action='store_false',
        help='Disable default forced alignment and sentence-level metric calculation.',
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
        '--sentence_output_dir',
        type=Path,
        default=Path('./output/sentence_level'),
        help='Folder for sentence-level acoustic feature outputs.',
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


def _find_transcript(transcript_dir: Path, file_id: str) -> Path | None:
    direct = transcript_dir / f'{file_id}.txt'
    if direct.exists():
        return direct
    matches = sorted(p for p in transcript_dir.rglob('*.txt') if p.stem == file_id)
    return matches[0] if matches else None


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _run_transcription(args: argparse.Namespace, wav_jobs: list[tuple[str, Path]], logger) -> Path:
    model_path = Path(args.asr_model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f'Qwen3-ASR model not found at {model_path}. Set --asr_model to a valid local model directory.'
        )

    args.transcription_output_dir.mkdir(parents=True, exist_ok=True)
    transcribe_script = Path(__file__).parent / 'src' / 'align' / 'transcribe.py'

    for file_id, wav_path in tqdm(wav_jobs, desc='Transcribing with Qwen3-ASR'):
        output_text = args.transcription_output_dir / f'{file_id}.txt'
        output_json = args.transcription_output_dir / f'{file_id}.qwen3_asr.json'
        cmd = [
            sys.executable,
            str(transcribe_script),
            '--audio',
            str(wav_path),
            '--model',
            args.asr_model,
            '--device-map',
            args.device_map,
            '--dtype',
            args.dtype,
            '--max-inference-batch-size',
            str(args.asr_max_inference_batch_size),
            '--max-new-tokens',
            str(args.asr_max_new_tokens),
            '--output-text',
            str(output_text),
            '--output-json',
            str(output_json),
        ]
        if args.asr_language:
            cmd.extend(['--language', args.asr_language])
        logger.info(f'[{file_id}] Running Qwen3-ASR transcription...')
        _run_command(cmd)
        logger.info(f'[{file_id}] Saved Qwen3-ASR transcript: {output_text}')

    return args.transcription_output_dir


def _run_forced_alignment(args: argparse.Namespace, wav_jobs: list[tuple[str, Path]], logger, cfg: dict) -> None:
    transcript_dir = args.transcript_dir or args.input_dir
    model_path = Path(args.forced_align_model).expanduser()
    if importlib.util.find_spec('qwen_asr') is None or importlib.util.find_spec('torch') is None:
        logger.warning(
            'Skip forced alignment and sentence-level metrics: qwen_asr/torch is not installed in this Python environment. '
            'Install/use the unified qwen3-asr-aligner environment or pass --no_forced_align.'
        )
        return
    if not model_path.exists():
        logger.warning(
            f'Skip forced alignment and sentence-level metrics: model not found at {model_path}. '
            'Pass --no_forced_align to silence this step, or set --forced_align_model to a valid model directory.'
        )
        return

    if args.run_transcription:
        try:
            transcript_dir = _run_transcription(args, wav_jobs, logger)
        except Exception:
            logger.error(f'Skip forced alignment: Qwen3-ASR transcription failed.\n{traceback.format_exc()}')
            return

    args.align_output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    args.sentence_output_dir.mkdir(parents=True, exist_ok=True)

    align_script = Path(__file__).parent / 'src' / 'align' / 'forced_align.py'
    metrics_script = Path(__file__).parent / 'src' / 'align' / 'metrics.py'

    for file_id, wav_path in tqdm(wav_jobs, desc='Forced aligning and sentence-level features'):
        transcript_path = _find_transcript(transcript_dir, file_id)
        if transcript_path is None:
            logger.warning(f'Skip forced alignment for {file_id}: transcript not found in {transcript_dir}')
            continue

        output_json = args.align_output_dir / f'{file_id}.qwen3_forced_align.json'
        output_tsv = args.align_output_dir / f'{file_id}.qwen3_forced_align.tsv'

        logger.info(f'[{file_id}] Running Qwen3 forced alignment...')
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
        logger.info(f'[{file_id}] Calculating global and sentence-level alignment metrics...')
        _run_command(metrics_cmd)

        for mode in ('independent_filler', 'merge_filler_to_next'):
            sentence_metrics_csv = args.metrics_output_dir / f'{output_json.stem}.{mode}.metrics.csv'
            if not sentence_metrics_csv.exists():
                continue
            sentence_output_csv = args.sentence_output_dir / f'{file_id}.{mode}.sentence_acoustic.csv'
            sentence_work_dir = args.work_dir / 'sentence_wav' / file_id / mode
            logger.info(f'[{file_id}] Extracting sentence-level acoustic features for {mode}...')
            extract_sentence_acoustic_features(
                sentence_metrics_csv,
                wav_path,
                cfg,
                sentence_work_dir,
                sentence_output_csv,
            )
            logger.info(f'[{file_id}] Saved sentence-level acoustic features: {sentence_output_csv}')


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
            logger.info(f'[{file_id}] Converting to standardized wav...')
            convert_to_wav(
                audio_path,
                wav_path,
                sample_rate=int(cfg.get('sample_rate', 16000)),
                mono=bool(cfg.get('mono', True)),
            )
            wav_jobs.append((file_id, wav_path))

            logger.info(f'[{file_id}] Extracting spectral summary features...')
            spectral_feats = extract_spectral_features(wav_path, cfg)
            logger.info(f'[{file_id}] Extracting openSMILE summary features...')
            smile_feats = extract_opensmile_features(wav_path, cfg)
            logger.info(f'[{file_id}] Extracting Praat summary features...')
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
        logger.info('Traditional acoustic extraction finished. Starting Qwen3-ForcedAligner and sentence-level metrics.')
        _run_forced_alignment(args, wav_jobs, logger, cfg)
    else:
        logger.info('Forced alignment and sentence-level metrics disabled by --no_forced_align.')


if __name__ == '__main__':
    main()
