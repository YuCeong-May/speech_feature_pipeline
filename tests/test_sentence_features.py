from pathlib import Path

from src.sentence_features import _prefixed_extract


def test_sentence_praat_excludes_whole_file_voice_quality_features():
    def extractor(_wav_path: Path, _cfg: dict) -> dict:
        return {
            "praat_F0_mean_hz": 120.0,
            "praat_hnr_mean_db": 20.0,
            "praat_jitter_local": 0.01,
            "praat_jitter_rap": 0.02,
            "praat_shimmer_local": 0.03,
            "praat_shimmer_apq3": 0.04,
        }

    result = _prefixed_extract("sentence_praat", extractor, Path("dummy.wav"), {})

    assert result == {"sentence_praat_F0_mean_hz": 120.0}
