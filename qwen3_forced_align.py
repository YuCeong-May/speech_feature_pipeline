#!/usr/bin/env python3
"""Run Qwen3-ForcedAligner on one audio/transcript pair."""

import argparse
import json
from pathlib import Path

import torch
from qwen_asr import Qwen3ForcedAligner


def read_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def result_to_dicts(result):
    rows = []
    for item in result:
        rows.append(
            {
                "text": item.text,
                "start_time": float(item.start_time),
                "end_time": float(item.end_time),
            }
        )
    return rows


def write_tsv(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("text\tstart_time\tend_time\n")
        for row in rows:
            text = str(row["text"]).replace("\t", " ").replace("\n", " ")
            f.write(f"{text}\t{row['start_time']:.3f}\t{row['end_time']:.3f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ForcedAligner timestamp alignment.")
    parser.add_argument("--audio", required=True, help="Input wav/mp3/flac path.")
    parser.add_argument("--text", required=True, help="Transcript text path.")
    parser.add_argument(
        "--model",
        default="../pre_trained_models/Qwen3-ForcedAligner-0.6B",
        help="Local Qwen3-ForcedAligner model directory.",
    )
    parser.add_argument("--language", default="Chinese", help='Language name, e.g. "Chinese" or "English".')
    parser.add_argument("--device-map", default="cuda:0", help='Device map, e.g. "cuda:0", "auto", or "cpu".')
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-tsv", default=None, help="Output TSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    text_path = Path(args.text).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_json = Path(args.output_json) if args.output_json else audio_path.with_suffix(".qwen3_forced_align.json")
    output_tsv = Path(args.output_tsv) if args.output_tsv else audio_path.with_suffix(".qwen3_forced_align.tsv")

    transcript = read_transcript(text_path)
    if not transcript:
        raise ValueError(f"Empty transcript: {text_path}")

    model = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=args.device_map,
    )

    results = model.align(
        audio=str(audio_path),
        text=transcript,
        language=args.language,
    )
    rows = result_to_dicts(results[0])

    payload = {
        "audio": str(audio_path),
        "text": str(text_path),
        "model": str(model_path),
        "language": args.language,
        "items": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(rows, output_tsv)

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote TSV:  {output_tsv}")
    print(f"Items: {len(rows)}")


if __name__ == "__main__":
    main()
