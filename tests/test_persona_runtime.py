"""Tests for MVP5 persona_runtime — system prompt extraction + emotion triggers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovtuber.pipeline.persona_runtime import (
    PersonaRuntime,
    build_persona_runtime,
    extract_emotion_triggers,
    save_persona_runtime,
    to_llm_system_prompt,
)


SAMPLE_MD = """## 基本資料
- 名字：testB
- 年齡：18
- 身高：158 cm
- 生日：4/15

## 個性詳細
- 話不多，每句都讓人想多想兩秒
- 對自己過去保留一份神秘
- 情緒表達極簡但不冷漠

## 簽名 Prop
撿到的舊耳機（左耳邊有個被刮掉的貼紙痕跡，露出原本的圖樣一角） — 每場直播都會出現，是觀眾辨認 testB 的關鍵記憶點。

## 背景故事
testB 是白天在霓虹招牌維修店學徒。

## 興趣與嗜好
- 收集老物件並記錄它們的「上一手故事」
- 看電影並寫長篇心得

## 口頭禪
- 「欸欸欸這個我也想試試！」
- 「⋯⋯讓我想一下喔」
- 「謝謝你陪我到這裡」

## 直播風格建議
適合做雜談。

## 與觀眾互動方式
稱呼粉絲為「testB的朋友」（不用「家人」這種大詞，保留適當距離感）。
互動語氣偏溫和、會記得常駐觀眾的暱稱。
"""


def test_to_llm_system_prompt_within_500_chars():
    prompt = to_llm_system_prompt(SAMPLE_MD)
    assert len(prompt) <= 500


def test_to_llm_system_prompt_includes_nickname_and_role():
    prompt = to_llm_system_prompt(SAMPLE_MD)
    assert "testB" in prompt
    assert "VTuber" in prompt


def test_to_llm_system_prompt_includes_signature_prop():
    prompt = to_llm_system_prompt(SAMPLE_MD)
    assert "舊耳機" in prompt


def test_to_llm_system_prompt_includes_audience_form_of_address():
    prompt = to_llm_system_prompt(SAMPLE_MD)
    assert "testB的朋友" in prompt


def test_to_llm_system_prompt_includes_anti_ai_rule():
    """system prompt 必須明確要 LLM 不要破壞角色 / 不承認自己是 AI。"""
    prompt = to_llm_system_prompt(SAMPLE_MD)
    assert "AI" in prompt or "ai" in prompt


def test_extract_emotion_triggers_has_defaults():
    triggers = extract_emotion_triggers(SAMPLE_MD)
    # 預設常見情緒
    assert "笑" in triggers and triggers["笑"] == "joy"
    assert "難過" in triggers and triggers["難過"] == "sorrow"
    assert "誒" in triggers and triggers["誒"] in ("surprised", "fun")


def test_extract_emotion_triggers_includes_persona_catchphrases():
    """從口頭禪自動加入角色獨特的觸發詞 → joy。"""
    triggers = extract_emotion_triggers(SAMPLE_MD)
    # 「欸欸欸這個我也想試試！」結尾 ! → joy trigger
    assert any("試試" in k for k in triggers.keys())
    # 「謝謝你陪我到這裡」 → joy
    assert any("謝謝" in k for k in triggers.keys())


def test_build_persona_runtime_structure():
    rt = build_persona_runtime(SAMPLE_MD)
    assert isinstance(rt, PersonaRuntime)
    assert rt.nickname == "testB"
    assert rt.signature_prop is not None and "舊耳機" in rt.signature_prop
    assert rt.audience_form_of_address == "testB的朋友"
    assert len(rt.catchphrases) >= 3
    assert rt.short_system_prompt
    assert rt.emotion_triggers
    assert "joy" in rt.emotion_triggers.values()


def test_save_persona_runtime_writes_valid_json(tmp_path: Path):
    rt = build_persona_runtime(SAMPLE_MD)
    out = tmp_path / "test_runtime.json"
    save_persona_runtime(rt, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["nickname"] == "testB"
    assert "舊耳機" in data["signature_prop"]
    assert isinstance(data["emotion_triggers"], dict)
    assert isinstance(data["catchphrases"], list)


def test_runtime_handles_missing_signature_prop_section():
    """若 markdown 沒簽名 prop 章節（老 persona），仍能跑通。"""
    md_no_prop = """## 基本資料
- 名字：oldChar

## 個性詳細
- 安靜

## 背景故事
平凡。

## 興趣與嗜好
- 散步

## 口頭禪
- 「謝謝」

## 直播風格建議
雜談。

## 與觀眾互動方式
稱呼粉絲為「夥伴」。
"""
    rt = build_persona_runtime(md_no_prop)
    assert rt.signature_prop is None or rt.signature_prop == ""
    # 仍然要能產出 system prompt 不爆
    assert len(rt.short_system_prompt) > 0
    assert "oldChar" in rt.short_system_prompt
