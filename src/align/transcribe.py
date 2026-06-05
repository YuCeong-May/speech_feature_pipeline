#!/usr/bin/env python3
"""Transcribe one audio file with Qwen3-ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from qwen_asr import Qwen3ASRModel


def _dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _value(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _serializable(obj: Any):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {
        key: value
        for key, value in vars(obj).items()
        if not key.startswith("_")
    } if hasattr(obj, "__dict__") else str(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Qwen3-ASR.")
    parser.add_argument("--audio", required=True, help="Input audio path.")
    parser.add_argument("--model", default="../pre_trained_models/Qwen3-ASR-1.7B", help="Local Qwen3-ASR model path.")
    parser.add_argument("--language", default=None, help="ASR language name. Leave unset for automatic language detection.")
    parser.add_argument("--device-map", default="cuda:0", help="Model device map, e.g. cuda:0, auto, or cpu.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output-text", required=True, help="Output transcript .txt path.")
    parser.add_argument("--output-json", default=None, help="Optional output metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_text = Path(args.output_text)
    output_json = Path(args.output_json) if args.output_json else output_text.with_suffix(".json")

    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=_dtype(args.dtype),
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(
        audio=str(audio_path),
        language=args.language,
    )
    if not results:
        raise RuntimeError(f"Qwen3-ASR returned no result for {audio_path}")

    first = results[0]
    text = str(_value(first, "text", "")).strip()
    if not text:
        raise RuntimeError(f"Qwen3-ASR returned empty transcript for {audio_path}")

    payload = {
        "audio": str(audio_path),
        "model": str(model_path),
        "requested_language": args.language,
        "detected_language": _value(first, "language"),
        "text": text,
        "raw_result": _serializable(first),
    }

    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_text.write_text(text + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote transcript: {output_text}")
    print(f"Wrote metadata:   {output_json}")
    print(f"Language: {_value(first, 'language')}")


if __name__ == "__main__":
    main()
