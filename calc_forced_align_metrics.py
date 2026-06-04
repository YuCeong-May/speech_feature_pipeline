#!/usr/bin/env python3
"""Calculate speech metrics from Qwen3-ForcedAligner output.

Two sentence policies are reported:
1. independent_filler: standalone filler lines such as "呃" are independent sentences.
2. merge_filler_to_next: standalone filler lines are merged into the next content line.
"""

import argparse
import csv
import json
import re
from pathlib import Path


FILLERS = {"呃", "嗯", "啊", "额", "呃嗯", "嗯嗯", "呃呃", "唔", "唔嗯"}


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_countable_char(ch: str) -> bool:
    return is_cjk(ch) or ch.isascii() and ch.isalnum()


def normalized_chars(text: str) -> list[str]:
    return [ch.lower() for ch in text if is_countable_char(ch)]


def visible_text(text: str) -> str:
    return " ".join(text.split())


def is_filler_line(line: str) -> bool:
    compact = "".join(ch for ch in line.strip() if is_cjk(ch))
    return compact in FILLERS


def drop_trailing_zero_duration(items: list[dict]) -> list[dict]:
    end = len(items)
    while end > 0 and float(items[end - 1]["end_time"]) <= float(items[end - 1]["start_time"]):
        end -= 1
    return items[:end]


def load_items(path: Path, drop_tail_zero: bool) -> list[dict]:
    if path.suffix.lower() == ".tsv":
        items = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                items.append(
                    {
                        "text": row["text"],
                        "start_time": float(row["start_time"]),
                        "end_time": float(row["end_time"]),
                    }
                )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["items"]
    if drop_tail_zero:
        items = drop_trailing_zero_duration(items)
    return [
        {
            "text": str(item["text"]),
            "start_time": float(item["start_time"]),
            "end_time": float(item["end_time"]),
        }
        for item in items
    ]


def load_transcript_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_sentence_lines(lines: list[str], mode: str) -> list[str]:
    if mode == "independent_filler":
        return lines
    if mode != "merge_filler_to_next":
        raise ValueError(f"Unknown mode: {mode}")

    merged = []
    pending_fillers = []
    for line in lines:
        if is_filler_line(line):
            pending_fillers.append(line)
            continue
        if pending_fillers:
            merged.append(" ".join(pending_fillers + [line]))
            pending_fillers = []
        else:
            merged.append(line)
    if pending_fillers:
        merged.append(" ".join(pending_fillers))
    return merged


def item_norm_text(item: dict) -> list[str]:
    return normalized_chars(item["text"])


def assign_items_to_sentences(sentence_lines: list[str], items: list[dict]) -> list[dict]:
    sentences = []
    item_idx = 0
    char_idx = 0

    for sentence_id, line in enumerate(sentence_lines, start=1):
        target_chars = normalized_chars(line)
        consumed = []
        matched_chars = []

        while char_idx < len(target_chars) and item_idx < len(items):
            item = items[item_idx]
            chars = item_norm_text(item)
            item_idx += 1
            if not chars:
                continue

            consumed.append(item)
            for ch in chars:
                if char_idx >= len(target_chars):
                    break
                # The aligner output is usually character-level. If a mismatch
                # appears, keep moving but record the text for later inspection.
                matched_chars.append(ch)
                char_idx += 1

        sentences.append(
            {
                "sentence_id": sentence_id,
                "text": visible_text(line),
                "target_char_count": len(target_chars),
                "matched_char_count": len(matched_chars),
                "missing_char_count": max(0, len(target_chars) - len(matched_chars)),
                "items": consumed,
            }
        )
        char_idx = 0

    return sentences


def count_chars(text: str) -> int:
    return sum(1 for ch in text if is_countable_char(ch))


def count_syllables(text: str) -> int:
    # For Mandarin speech, one Chinese character is treated as one syllable.
    # English/number runs are counted as one token each as a conservative fallback.
    cjk_count = sum(1 for ch in text if is_cjk(ch))
    non_cjk_runs = re.findall(r"[A-Za-z0-9]+", text)
    return cjk_count + len(non_cjk_runs)


def count_words(text: str) -> int:
    # Chinese characters are counted individually; English/number runs are
    # counted as one word each.
    cjk_count = sum(1 for ch in text if is_cjk(ch))
    non_cjk_runs = re.findall(r"[A-Za-z0-9]+", text)
    return cjk_count + len(non_cjk_runs)


def count_ascii_chars(text: str) -> int:
    return sum(1 for ch in text if ch.isascii() and ch.isalnum())


def count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if is_cjk(ch))


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def sentence_metrics(sentence: dict, pause_threshold: float) -> dict:
    items = sentence["items"]
    text = sentence["text"]
    valid_items = [item for item in items if item["end_time"] >= item["start_time"]]

    if valid_items:
        start = min(item["start_time"] for item in valid_items)
        end = max(item["end_time"] for item in valid_items)
    else:
        start = 0.0
        end = 0.0

    total_duration = max(0.0, end - start)
    speech_time = sum(max(0.0, item["end_time"] - item["start_time"]) for item in valid_items)

    pause_durations = []
    ordered = sorted(valid_items, key=lambda x: (x["start_time"], x["end_time"]))
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur["start_time"] - prev["end_time"]
        if gap >= pause_threshold:
            pause_durations.append(gap)

    pause_time = sum(pause_durations)
    char_count = count_chars(text)
    word_count = count_words(text)
    syllable_count = count_syllables(text)

    return {
        "sentence_id": sentence["sentence_id"],
        "text": text,
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "total_duration": round(total_duration, 3),
        "speech_time": round(speech_time, 3),
        "pause_time": round(pause_time, 3),
        "pause_count": len(pause_durations),
        "pause_ratio": round(safe_div(pause_time, total_duration), 6),
        "word_count": word_count,
        "char_count": char_count,
        "syllable_count": syllable_count,
        "speech_rate_words_per_sec": round(safe_div(word_count, total_duration), 6),
        "speech_rate_chars_per_sec": round(safe_div(char_count, total_duration), 6),
        "speech_rate_words_per_min": round(safe_div(word_count, total_duration) * 60, 6),
        "speech_rate_chars_per_min": round(safe_div(char_count, total_duration) * 60, 6),
        "avg_syllable_duration": round(safe_div(speech_time, syllable_count), 6),
        "target_char_count": sentence["target_char_count"],
        "matched_char_count": sentence["matched_char_count"],
        "missing_char_count": sentence["missing_char_count"],
    }


def summary_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    nonempty = [row for row in rows if row["total_duration"] > 0]
    if nonempty:
        start = min(row["start_time"] for row in nonempty)
        end = max(row["end_time"] for row in nonempty)
    else:
        start = 0.0
        end = 0.0

    total_duration = max(0.0, end - start)
    speech_time = sum(row["speech_time"] for row in rows)
    pause_time = sum(row["pause_time"] for row in rows)
    word_count = sum(row["word_count"] for row in rows)
    char_count = sum(row["char_count"] for row in rows)
    syllable_count = sum(row["syllable_count"] for row in rows)

    return {
        "sentence_count": len(rows),
        "start_time": round(start, 3),
        "end_time": round(end, 3),
        "total_duration": round(total_duration, 3),
        "speech_time": round(speech_time, 3),
        "pause_time": round(pause_time, 3),
        "pause_count": sum(row["pause_count"] for row in rows),
        "pause_ratio": round(safe_div(pause_time, total_duration), 6),
        "word_count": word_count,
        "char_count": char_count,
        "syllable_count": syllable_count,
        "speech_rate_words_per_sec": round(safe_div(word_count, total_duration), 6),
        "speech_rate_chars_per_sec": round(safe_div(char_count, total_duration), 6),
        "speech_rate_words_per_min": round(safe_div(word_count, total_duration) * 60, 6),
        "speech_rate_chars_per_min": round(safe_div(char_count, total_duration) * 60, 6),
        "avg_syllable_duration": round(safe_div(speech_time, syllable_count), 6),
    }


def global_summary_metrics(items: list[dict], pause_threshold: float) -> dict:
    if not items:
        return {}

    ordered = sorted(items, key=lambda x: (x["start_time"], x["end_time"]))
    start = min(item["start_time"] for item in ordered)
    end = max(item["end_time"] for item in ordered)
    total_duration = max(0.0, end - start)
    speech_time = sum(max(0.0, item["end_time"] - item["start_time"]) for item in ordered)

    threshold_gaps = []
    all_positive_gaps = []
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur["start_time"] - prev["end_time"]
        if gap > 0:
            all_positive_gaps.append(gap)
        if gap >= pause_threshold:
            threshold_gaps.append(gap)

    pause_time = sum(threshold_gaps)
    text = "".join(str(item["text"]) for item in ordered)
    cjk_char_count = count_cjk_chars(text)
    ascii_char_count = count_ascii_chars(text)
    word_count = count_words(text)
    char_count = cjk_char_count + ascii_char_count
    syllable_count = count_syllables(text)

    return {
        "total_duration_with_pauses_sec": round(total_duration, 3),
        "speech_time_sec": round(speech_time, 3),
        "pause_threshold_sec": pause_threshold,
        "pause_time_sec": round(pause_time, 3),
        "pause_count": len(threshold_gaps),
        "pause_ratio_percent": round(safe_div(pause_time, total_duration) * 100, 6),
        "all_gap_time_sec_no_threshold": round(sum(all_positive_gaps), 3),
        "all_positive_gap_count_no_threshold": len(all_positive_gaps),
        "word_count": word_count,
        "char_count": char_count,
        "cjk_char_count": cjk_char_count,
        "ascii_char_count": ascii_char_count,
        "syllable_count": syllable_count,
        "speech_rate_words_per_sec": round(safe_div(word_count, total_duration), 6),
        "speech_rate_chars_per_sec": round(safe_div(char_count, total_duration), 6),
        "speech_rate_words_per_min": round(safe_div(word_count, total_duration) * 60, 6),
        "speech_rate_chars_per_min": round(safe_div(char_count, total_duration) * 60, 6),
        "avg_syllable_duration_sec": round(safe_div(speech_time, syllable_count), 6),
        "first_valid_start_sec": round(start, 3),
        "last_valid_end_sec": round(end, 3),
        "valid_token_count": len(ordered),
        "valid_text": text,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate forced-alignment speech metrics.")
    parser.add_argument(
        "--align-file",
        "--align-json",
        dest="align_json",
        required=True,
        help="Qwen3 forced alignment file, JSON or TSV.",
    )
    parser.add_argument("--transcript", default=None, help="Original transcript text. Required for sentence CSVs.")
    parser.add_argument("--pause-threshold", type=float, default=0.2, help="Pause threshold in seconds.")
    parser.add_argument(
        "--keep-trailing-zero-duration",
        action="store_true",
        help="Keep trailing zero-duration alignment tokens. By default they are dropped.",
    )
    parser.add_argument("--output-json", default=None, help="Output metrics JSON.")
    parser.add_argument("--output-dir", default=None, help="Directory for per-version CSV files.")
    return parser.parse_args()


def main():
    args = parse_args()
    align_path = Path(args.align_json).expanduser().resolve()
    transcript_path = Path(args.transcript).expanduser().resolve() if args.transcript else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else align_path.parent
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else output_dir / f"{align_path.stem}.metrics.json"
    )

    items = load_items(align_path, drop_tail_zero=not args.keep_trailing_zero_duration)
    summary_csv = output_dir / f"{align_path.stem}.summary.metrics.csv"

    payload = {
        "align_json": str(align_path),
        "transcript": str(transcript_path) if transcript_path else None,
        "pause_threshold": args.pause_threshold,
        "trailing_zero_duration_policy": "keep" if args.keep_trailing_zero_duration else "drop",
        "word_count_method": "cjk_char_plus_ascii_runs",
        "global_summary": global_summary_metrics(items, args.pause_threshold),
        "versions": {},
    }
    write_summary_csv(payload["global_summary"], summary_csv)

    if transcript_path:
        transcript_lines = load_transcript_lines(transcript_path)
        for mode in ("independent_filler", "merge_filler_to_next"):
            sentence_lines = build_sentence_lines(transcript_lines, mode)
            assigned = assign_items_to_sentences(sentence_lines, items)
            rows = [sentence_metrics(sentence, args.pause_threshold) for sentence in assigned]
            payload["versions"][mode] = {
                "summary": summary_metrics(rows),
                "sentences": rows,
            }
            write_csv(rows, output_dir / f"{align_path.stem}.{mode}.metrics.csv")

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote CSV:  {summary_csv}")
    if transcript_path:
        print(f"Wrote CSV:  {output_dir / (align_path.stem + '.independent_filler.metrics.csv')}")
        print(f"Wrote CSV:  {output_dir / (align_path.stem + '.merge_filler_to_next.metrics.csv')}")


if __name__ == "__main__":
    main()
