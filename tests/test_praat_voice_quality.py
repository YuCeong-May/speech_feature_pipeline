from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from src.extract_praat import extract_praat_features
from src.merge_features import chinese_name_for_feature


def test_extract_praat_voice_quality_fields():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 120 * t)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "tone.wav"
        sf.write(wav_path, y, sr)
        features = extract_praat_features(wav_path, {})

    assert "praat_hnr_mean_db" in features
    assert "praat_jitter_local" in features
    assert "praat_shimmer_local" in features
    assert np.isfinite(features["praat_hnr_mean_db"])
    assert np.isfinite(features["praat_jitter_local"])
    assert np.isfinite(features["praat_shimmer_local"])


def test_voice_quality_feature_name_mapping():
    assert chinese_name_for_feature("praat_hnr_mean_db") == "Praat谐波噪声比HNR均值_dB"
    assert chinese_name_for_feature("praat_jitter_local") == "Praat频率微扰Jitter_local"
    assert chinese_name_for_feature("praat_shimmer_local") == "Praat振幅微扰Shimmer_local"
