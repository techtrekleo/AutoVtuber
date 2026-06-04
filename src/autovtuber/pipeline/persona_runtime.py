"""MVP5 — 把 persona markdown 萃取成「runtime 用」結構化資料。

兩個輸出：
    1. `to_llm_system_prompt(md)` — ≤500 字濃縮版，給下游 chat runtime 當 system prompt
       （Open-LLM-VTuber 借鑑：persona = LLM 角色扮演 system prompt）
    2. `extract_emotion_triggers(md)` — { chinese_keyword: vrm_blendshape_preset_name }
       （從口頭禪/個性章節抽常見情緒詞，未來給 runtime 對應 blendshape）

存檔：`output/<basename>_persona_runtime.json`（與 persona.md 並排）
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.logging_setup import get_logger

_log = get_logger(__name__)


# ARKit / VRM 標準 blendshape preset 名（VSeeFace / VTube Studio 都認）
_VRM_PRESETS = {
    "joy", "angry", "sorrow", "fun", "surprised",
    "neutral", "blink", "blink_l", "blink_r",
}


# 中文情緒/觸發字典 → VRM blendshape preset（多個 keyword 可對同一 preset）
_DEFAULT_EMOTION_TRIGGERS: dict[str, str] = {
    # JOY（笑/開心）
    "笑": "joy", "哈哈": "joy", "開心": "joy", "好棒": "joy", "讚": "joy",
    "喜歡": "joy", "謝謝": "joy", "感動": "joy",
    # ANGRY（生氣）
    "生氣": "angry", "氣": "angry", "煩": "angry", "哼": "angry",
    "受不了": "angry", "夠了": "angry",
    # SORROW（難過）
    "難過": "sorrow", "嗚": "sorrow", "對不起": "sorrow", "唉": "sorrow",
    "失望": "sorrow", "孤單": "sorrow",
    # FUN（玩心/打趣）
    "嘿嘿": "fun", "欸": "fun", "玩": "fun", "打鬧": "fun",
    "好玩": "fun", "有趣": "fun",
    # SURPRISED（驚訝）
    "誒": "surprised", "哇": "surprised", "什麼": "surprised", "天哪": "surprised",
    "真的嗎": "surprised", "不會吧": "surprised",
    # NEUTRAL（無情緒）
    "嗯": "neutral", "好": "neutral", "了解": "neutral",
}


@dataclass
class PersonaRuntime:
    """給下游 chat runtime 用的結構化 persona。"""
    nickname: str
    short_system_prompt: str           # ≤500 字 system prompt，餵給 chat LLM
    catchphrases: list[str]            # 直接拿來訓練 LLM 的 few-shot
    signature_prop: str | None         # 簽名物件（可在 chat 中引用）
    audience_form_of_address: str | None  # 怎麼稱呼觀眾（如「testB的朋友」）
    emotion_triggers: dict[str, str]   # 中文觸發詞 → VRM blendshape preset


# ---------------- 萃取邏輯 ---------------- #


def _extract_section(md: str, heading: str) -> str:
    """從 markdown 抓 `## heading` 之後到下個 `## ` 之前的文字。"""
    pattern = rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, md, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_bullet_list(section_text: str) -> list[str]:
    """從章節內容抓 bullet list 條目（- 開頭）。"""
    items = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            # 移除註解（# ... 之後）
            content = re.split(r"\s+#\s+", content)[0].strip()
            # 移除 markdown 引號
            content = content.strip("「」")
            if content:
                items.append(content)
    return items


def _truncate_smart(text: str, max_chars: int) -> str:
    """截斷文字到 max_chars，盡量在句號/換行處斷。"""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # 嘗試在最近的句號/換行處斷
    for sep in ("。\n", "。", "\n", "，"):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.7:
            return cut[: idx + len(sep)].rstrip()
    return cut.rstrip() + "…"


def to_llm_system_prompt(md: str, max_chars: int = 500) -> str:
    """把七章節 persona markdown 濃縮成 ≤max_chars 的 chat system prompt。

    包含：身份 / 個性核心 / 簽名 prop / 稱呼觀眾 / 互動語氣 / 1-2 個口頭禪範例。
    """
    basic = _extract_section(md, "基本資料")
    personality = _extract_section(md, "個性詳細")
    sig_prop_section = _extract_section(md, "簽名 Prop")
    catchphrases_section = _extract_section(md, "口頭禪")
    interact = _extract_section(md, "與觀眾互動方式")

    # 名字
    nick_match = re.search(r"名字\s*[：:]\s*(.+)", basic)
    nick = nick_match.group(1).strip() if nick_match else "VTuber"

    # 個性精簡：取前 2 條 bullet
    personality_bullets = _extract_bullet_list(personality)
    personality_summary = "；".join(personality_bullets[:2]) if personality_bullets else "個性溫和"

    # 簽名 prop（取「—」前的物件名）
    sig_prop = ""
    if sig_prop_section:
        sig_prop = sig_prop_section.split("—")[0].strip()

    # 口頭禪：取前 2 個
    catchphrases = _extract_bullet_list(catchphrases_section)
    catchphrase_sample = "／".join(catchphrases[:2]) if catchphrases else ""

    # 觀眾稱呼
    address_match = re.search(r"稱呼粉絲為\s*[「『]([^」』]+)[」』]", interact)
    addr = address_match.group(1).strip() if address_match else "觀眾"

    # 互動語氣摘要：第一句完整話
    interact_first = re.split(r"[。\n]", interact)[0].strip()

    parts = [
        f"你是 VTuber「{nick}」，正在跟觀眾即時互動。",
        f"個性：{personality_summary}。",
    ]
    if sig_prop:
        parts.append(f"你有一個簽名物件：{sig_prop}，可在對話中自然提及。")
    parts.append(f"稱呼觀眾為「{addr}」。")
    if interact_first:
        parts.append(f"互動風格：{interact_first}。")
    if catchphrase_sample:
        parts.append(f"常用口頭禪：{catchphrase_sample}。")
    parts.append("回覆要短（2-3 句），保持角色一致，不要破壞設定也不要承認自己是 AI。")

    result = " ".join(parts)
    return _truncate_smart(result, max_chars)


def extract_emotion_triggers(
    md: str,
    base_triggers: dict[str, str] | None = None,
) -> dict[str, str]:
    """從 persona 抓「角色特有」的情緒觸發詞，跟 base 字典合併。

    策略：
        - 預設用 _DEFAULT_EMOTION_TRIGGERS 當 base
        - 從口頭禪章節再加入這個角色獨特的開場/結尾用詞 → 對應 "joy"
        - 從個性詳細若提到「情緒外顯」「藏不住」等 → 全開預設
    """
    base = dict(base_triggers if base_triggers is not None else _DEFAULT_EMOTION_TRIGGERS)
    catchphrases_section = _extract_section(md, "口頭禪")
    catchphrases = _extract_bullet_list(catchphrases_section)
    # 把「！」或「謝謝」結尾的口頭禪當 joy trigger（去除引號 + 去 prop 綁定段）
    for phrase in catchphrases:
        clean = phrase.split("（")[0].strip().strip("「」")
        if not clean or len(clean) > 20:
            continue
        if clean.endswith(("！", "!")):
            base.setdefault(clean, "joy")
        elif "謝謝" in clean or "感動" in clean:
            base.setdefault(clean, "joy")
        elif "對不起" in clean or "難過" in clean:
            base.setdefault(clean, "sorrow")
    return base


def build_persona_runtime(md: str) -> PersonaRuntime:
    """從 markdown 建立 PersonaRuntime。"""
    basic = _extract_section(md, "基本資料")
    nick_match = re.search(r"名字\s*[：:]\s*(.+)", basic)
    nick = nick_match.group(1).strip() if nick_match else "VTuber"

    catchphrases_section = _extract_section(md, "口頭禪")
    catchphrases = _extract_bullet_list(catchphrases_section)[:6]

    sig_prop_section = _extract_section(md, "簽名 Prop")
    sig_prop = sig_prop_section.split("—")[0].strip() if sig_prop_section else None

    interact = _extract_section(md, "與觀眾互動方式")
    address_match = re.search(r"稱呼粉絲為\s*[「『]([^」』]+)[」』]", interact)
    addr = address_match.group(1).strip() if address_match else None

    return PersonaRuntime(
        nickname=nick,
        short_system_prompt=to_llm_system_prompt(md),
        catchphrases=catchphrases,
        signature_prop=sig_prop,
        audience_form_of_address=addr,
        emotion_triggers=extract_emotion_triggers(md),
    )


def save_persona_runtime(runtime: PersonaRuntime, path: Path) -> Path:
    """寫 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(runtime), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log.info("📦 Persona runtime JSON → {}", path)
    return path
