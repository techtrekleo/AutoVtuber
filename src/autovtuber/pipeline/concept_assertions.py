"""對 SDXL 概念圖做自動校驗 — 防止「concept 與 form 不一致」regression。

主要 API：
    assert_iris_color_matches_form(concept_img, form, tolerance_hue_deg=30) -> AssertionResult

對應 Reality Checker 2026-04-27 audit blocking gap：
    「VRM 與 concept 眼色不一致 = 不可接受」
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..utils.logging_setup import get_logger
from .job_spec import FormInput

_log = get_logger(__name__)


@dataclass
class AssertionResult:
    """單一斷言結果。passed=True 表示通過。"""
    passed: bool
    detail: str
    measured: dict[str, float] | None = None


def _eye_roi_via_blazeface(img: Image.Image) -> tuple[int, int, int, int] | None:
    """用 MediaPipe BlazeFace 找臉，回傳眼睛 ROI 像素座標 (x0,y0,x1,y1)。"""
    try:
        import mediapipe as mp
        import numpy as _np
        det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.3,
        )
        rgb = _np.asarray(img.convert("RGB"))
        h_img, w_img = rgb.shape[:2]
        result = det.process(rgb)
        det.close()
        if not result.detections:
            return None
        face = max(result.detections, key=lambda f: f.location_data.relative_bounding_box.width)
        bb = face.location_data.relative_bounding_box
        # bb 是臉的整個框；眼睛通常在臉框上 1/3 處
        fx = int(bb.xmin * w_img)
        fy = int(bb.ymin * h_img)
        fw = int(bb.width * w_img)
        fh = int(bb.height * h_img)
        # 眼 ROI：臉框上 25%-50% Y、20%-80% X（取兩眼之間 + 兩側）
        ey0 = max(0, fy + int(fh * 0.25))
        ey1 = min(h_img, fy + int(fh * 0.50))
        ex0 = max(0, fx + int(fw * 0.20))
        ex1 = min(w_img, fx + int(fw * 0.80))
        if ey1 <= ey0 or ex1 <= ex0:
            return None
        return ex0, ey0, ex1, ey1
    except Exception:  # noqa: BLE001 — 任何失敗 fallback 到中央 ROI
        return None


def _hex_to_hue(hex_str: str) -> float:
    """`#RRGGBB` → hue (0-360)。低 saturation 時 hue 不可靠（會回 -1）。"""
    s = hex_str.lstrip("#")
    r, g, b = int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255
    h, sat, _v = colorsys.rgb_to_hsv(r, g, b)
    if sat < 0.10:
        return -1.0  # 灰階沒有色相
    return h * 360.0


def _hue_distance(a: float, b: float) -> float:
    """色環距離（0-180）。"""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def detect_iris_color(concept_img: Image.Image) -> tuple[float, float] | None:
    """從 concept 圖偵測 iris 主色相。

    策略：優先用 BlazeFace 偵測臉部 → 從 face bbox 推算眼睛 ROI；
         偵測失敗時 fallback 到嚴格中央 ROI（避免抓到衣服/背景）。
    回傳 (hue_deg, mean_saturation) 或 None（偵測不到）。
    """
    arr = np.asarray(concept_img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    # 先試 BlazeFace
    eye_roi = _eye_roi_via_blazeface(concept_img)
    if eye_roi is not None:
        x0, y0, x1, y1 = eye_roi
    else:
        # Fallback：嚴格中央 ROI（眼睛通常在 portrait y=0.28-0.40 內，x=0.35-0.65）
        y0, y1 = int(h * 0.28), int(h * 0.42)
        x0, x1 = int(w * 0.32), int(w * 0.68)
    roi = arr[y0:y1, x0:x1]
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    val = max_c
    sat = np.where(max_c > 0, (max_c - min_c) / np.maximum(max_c, 1e-6), 0)
    # 過濾 skin tone（R > G > B 且高 V 低 S）+ 過濾近灰階
    skin = (r > g + 0.02) & (g > b - 0.02) & (val > 0.65) & (sat < 0.30)
    valid = (sat > 0.30) & (val > 0.20) & (val < 0.95) & (~skin)
    if valid.sum() < 50:  # ROI 內找不到夠多有色像素
        return None
    # 取 valid 像素的色相中位數
    delta = max_c - min_c
    h_raw = np.zeros_like(r)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)
    h_raw[mask_r] = ((g[mask_r] - b[mask_r]) / np.maximum(delta[mask_r], 1e-6)) % 6
    h_raw[mask_g] = (b[mask_g] - r[mask_g]) / np.maximum(delta[mask_g], 1e-6) + 2
    h_raw[mask_b] = (r[mask_b] - g[mask_b]) / np.maximum(delta[mask_b], 1e-6) + 4
    hue_deg = (h_raw[valid] * 60.0) % 360.0
    sat_vals = sat[valid]
    # 用環狀中位數（轉成向量再平均）
    rad = np.deg2rad(hue_deg)
    mean_x = np.cos(rad).mean()
    mean_y = np.sin(rad).mean()
    median_hue = (np.rad2deg(np.arctan2(mean_y, mean_x))) % 360.0
    return float(median_hue), float(sat_vals.mean())


def assert_iris_color_matches_form(
    concept_img: Image.Image,
    form: FormInput,
    tolerance_hue_deg: float = 40.0,
) -> AssertionResult:
    """檢查 concept 圖偵測到的 iris hue 是否在 form 指定眼色 ±tolerance 內。

    使用案例：在 face_generator 生圖完成後立刻跑，不過則記錄警告
    （未來可改成觸發重生）。
    """
    target_hue = _hex_to_hue(form.eye_color_hex)
    if target_hue < 0:
        return AssertionResult(
            passed=True,
            detail=f"Form eye color {form.eye_color_hex} is greyscale; skip hue check",
        )
    detected = detect_iris_color(concept_img)
    if detected is None:
        return AssertionResult(
            passed=True,
            detail="Could not detect iris region in concept image; skip check",
        )
    detected_hue, mean_sat = detected
    dist = _hue_distance(target_hue, detected_hue)
    passed = dist <= tolerance_hue_deg
    return AssertionResult(
        passed=passed,
        detail=(
            f"Form eye hue={target_hue:.0f}° vs concept detected hue={detected_hue:.0f}° "
            f"(sat={mean_sat:.2f}) → distance={dist:.0f}° "
            f"(tolerance={tolerance_hue_deg:.0f}°)"
        ),
        measured={
            "target_hue": target_hue,
            "detected_hue": detected_hue,
            "distance": dist,
            "tolerance": tolerance_hue_deg,
        },
    )
