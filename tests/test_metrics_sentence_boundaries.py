import src.align.metrics as metrics
from src.align.metrics import add_inter_sentence_gaps, assign_items_to_sentences, build_sentence_lines, sentence_metrics, split_transcript_lines


def _item(text, start, end):
    return {"text": text, "start_time": start, "end_time": end}


def test_split_transcript_lines_preserves_punctuation_boundaries():
    assert split_transcript_lines(["呃，如果开心。嗯，继续说！最后一句"]) == [
        "呃，",
        "如果开心。",
        "嗯，",
        "继续说！",
        "最后一句",
    ]


def test_merge_filler_keeps_normal_sentence_boundaries():
    lines = split_transcript_lines(["呃，如果开心。嗯，继续说！"])
    assert build_sentence_lines(lines, "merge_filler_to_next") == [
        "呃， 如果开心。",
        "嗯， 继续说！",
    ]


def test_inter_sentence_gap_survives_merge_filler_to_next():
    lines = build_sentence_lines(split_transcript_lines(["呃，如果开心。嗯，继续说！"]), "merge_filler_to_next")
    items = [
        _item("呃", 0.0, 0.1),
        _item("如", 0.4, 0.5),
        _item("果", 0.5, 0.6),
        _item("开", 0.6, 0.7),
        _item("心", 0.7, 0.8),
        _item("嗯", 1.4, 1.5),
        _item("继", 1.8, 1.9),
        _item("续", 1.9, 2.0),
        _item("说", 2.0, 2.1),
    ]
    assigned = assign_items_to_sentences(lines, items)
    rows = [sentence_metrics(sentence, 0.2) for sentence in assigned]
    add_inter_sentence_gaps(rows, 0.2)

    assert len(rows) == 2
    assert rows[0]["pause_time"] == 0.3
    assert rows[1]["inter_sentence_gap_from_prev_sec"] == 0.6
    assert rows[1]["inter_sentence_pause_from_prev_sec"] == 0.6
    assert rows[1]["inter_sentence_pause_from_prev_count"] == 1


def test_jieba_word_pause_metrics_separate_within_and_between_word(monkeypatch):
    class FakeJieba:
        @staticmethod
        def lcut(text):
            assert text == "我觉得好"
            return ["我", "觉得", "好"]

    monkeypatch.setattr(metrics, "jieba", FakeJieba)
    sentence = {
        "sentence_id": 1,
        "text": "我觉得好",
        "target_char_count": 4,
        "matched_char_count": 4,
        "missing_char_count": 0,
        "items": [
            _item("我", 0.0, 0.1),
            _item("觉", 0.2, 0.3),
            _item("得", 0.7, 0.8),
            _item("好", 1.2, 1.3),
        ],
    }

    row = sentence_metrics(sentence, 0.2)

    assert row["pause_time"] == 0.8
    assert row["pause_count"] == 2
    assert row["within_word_pause_time"] == 0.4
    assert row["within_word_pause_count"] == 1
    assert row["between_word_pause_time"] == 0.4
    assert row["between_word_pause_count"] == 1
