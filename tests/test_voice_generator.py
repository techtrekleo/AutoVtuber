"""Tests for MVP5.5 VoiceGenerator — mocked VoxCPM, no actual model load."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autovtuber.pipeline.job_spec import (
    EyeShape, FormInput, HairLength, HairStyle, Personality, StyleGenre,
)
from autovtuber.pipeline.persona_runtime import PersonaRuntime
from autovtuber.pipeline.voice_generator import (
    VoiceGenerator,
    VoiceGenResult,
    _personality_to_voice_description,
    _pick_sample_text,
)


def make_form(personality=Personality.MYSTERIOUS_COOL):
    return FormInput(
        nickname="testB",
        hair_color_hex="#1E1E1E",
        hair_length=HairLength.MEDIUM,
        hair_style=HairStyle.WAVY,
        eye_color_hex="#7B1F1F",
        eye_shape=EyeShape.ALMOND,
        style=StyleGenre.ANIME_MODERN,
        personality=personality,
        base_model_id="AvatarSample_A",
    )


def make_runtime():
    return PersonaRuntime(
        nickname="testB",
        short_system_prompt="你是 testB。",
        catchphrases=["欸欸欸這個我也想試試！", "讓我想一下喔"],
        signature_prop="撿到的舊耳機",
        audience_form_of_address="testB的朋友",
        emotion_triggers={"笑": "joy"},
    )


# ---------------- Voice description ---------------- #


def test_personality_to_voice_description_includes_female_and_tone():
    form = make_form(Personality.MYSTERIOUS_COOL)
    rt = make_runtime()
    desc = _personality_to_voice_description(form, rt)
    assert desc.startswith("(") and desc.endswith(")")
    assert "young female" in desc
    # MYSTERIOUS_COOL → "low, calm, slightly raspy"
    assert "raspy" in desc or "slow" in desc or "calm" in desc


def test_personality_voice_descriptions_all_personalities():
    """全部 16 個 personality 都有 mapping。"""
    rt = make_runtime()
    for p in Personality:
        form = make_form(p)
        desc = _personality_to_voice_description(form, rt)
        assert len(desc) > 10
        assert "(" in desc and ")" in desc


# ---------------- Sample text picker ---------------- #


def test_pick_sample_text_prefers_first_catchphrase():
    rt = make_runtime()
    text = _pick_sample_text(rt)
    assert text == "欸欸欸這個我也想試試！"


def test_pick_sample_text_falls_back_when_no_catchphrases():
    rt = PersonaRuntime(
        nickname="silent",
        short_system_prompt="",
        catchphrases=[],
        signature_prop=None,
        audience_form_of_address=None,
        emotion_triggers={},
    )
    text = _pick_sample_text(rt)
    assert "silent" in text
    assert len(text) > 0


def test_pick_sample_text_strips_quotation_marks_and_comments():
    rt = PersonaRuntime(
        nickname="x",
        short_system_prompt="",
        catchphrases=["「你好啊朋友」  # 開場白"],
        signature_prop=None,
        audience_form_of_address=None,
        emotion_triggers={},
    )
    text = _pick_sample_text(rt)
    assert "「" not in text
    assert "#" not in text
    assert "你好啊朋友" in text


# ---------------- Generator (mocked VoxCPM) ---------------- #


def test_voice_generator_returns_result_when_voxcpm_succeeds(tmp_path: Path):
    """成功路徑：mock VoxCPM 回 1 秒 16kHz 假音 → 結果有 wav 路徑。"""
    fake_loader = MagicMock()

    # Context manager that yields a mock model
    fake_model = MagicMock()
    fake_model.tts_model.sample_rate = 16000
    fake_model.generate.return_value = np.zeros(16000, dtype=np.float32)  # 1s silence

    class FakeCtx:
        def __enter__(self_inner): return fake_model
        def __exit__(self_inner, *a): return False
    fake_loader.acquire.return_value = FakeCtx()

    fake_guard = MagicMock()
    fake_guard.check_or_raise = MagicMock(return_value=None)

    vg = VoiceGenerator(fake_loader, fake_guard)
    out_path = tmp_path / "test_voice.wav"
    result = vg.generate(make_form(), make_runtime(), out_path)

    assert result.wav_path == out_path
    assert out_path.exists()
    assert result.sample_rate == 16000
    assert "young female" in result.voice_description


def test_voice_generator_returns_none_path_when_voxcpm_crashes(tmp_path: Path):
    """失敗路徑：mock VoxCPM 拋例外 → 結果 wav_path None，不會拋出。"""
    fake_loader = MagicMock()

    class FakeCtx:
        def __enter__(self_inner): raise RuntimeError("voxcpm crashed")
        def __exit__(self_inner, *a): return False
    fake_loader.acquire.return_value = FakeCtx()

    fake_guard = MagicMock()
    fake_guard.check_or_raise = MagicMock(return_value=None)

    vg = VoiceGenerator(fake_loader, fake_guard)
    out_path = tmp_path / "test_voice.wav"
    result = vg.generate(make_form(), make_runtime(), out_path)

    # 不拋例外
    assert result.wav_path is None
    assert result.voice_description.startswith("(")
    # 不會寫檔
    assert not out_path.exists()


def test_voice_generator_returns_none_on_safety_abort(tmp_path: Path):
    """SafetyAbort 路徑：HardwareGuard 中止 → 跳過，不阻 pipeline。"""
    from autovtuber.safety.exceptions import SafetyAbort

    fake_loader = MagicMock()

    class FakeCtx:
        def __enter__(self_inner): raise SafetyAbort("VRAM exceeded mid-load")
        def __exit__(self_inner, *a): return False
    fake_loader.acquire.return_value = FakeCtx()

    fake_guard = MagicMock()
    fake_guard.check_or_raise = MagicMock(return_value=None)

    vg = VoiceGenerator(fake_loader, fake_guard)
    out_path = tmp_path / "test_voice.wav"
    result = vg.generate(make_form(), make_runtime(), out_path)

    assert result.wav_path is None
    assert not out_path.exists()


# ---------------- PersonaRuntime voice_profile field ---------------- #


def test_persona_runtime_voice_profile_default_none():
    """新增的 voice_profile 欄位預設 None，不破壞既有測試。"""
    rt = make_runtime()
    assert rt.voice_profile is None


def test_persona_runtime_voice_profile_settable():
    rt = make_runtime()
    rt.voice_profile = "(young female, calm, slow)"
    assert rt.voice_profile == "(young female, calm, slow)"
