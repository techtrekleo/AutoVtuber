"""Human-facing VTuber quality gate.

This module checks whether a generated VRM is meaningfully usable as a VTuber,
not merely whether it is a syntactically valid GLB/VRM file.

The checks are intentionally conservative and structural:
    - VRM extension exists
    - humanoid bone mapping is present
    - face/lip/blink/emotion blendshape clips exist
    - at least one skinned mesh exists
    - important texture slots are present and decodable when an atlas map is given

It does not try to judge beauty or art direction. It catches outputs that a human
would describe as "not a VTuber yet": broken mesh, no humanoid rig, no mouth
shapes, no blink, or missing face textures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ..vrm.texture_atlas import AtlasMap

_log = get_logger(__name__)


REQUIRED_HUMANOID_BONES = {
    "hips",
    "spine",
    "head",
    "neck",
    "leftUpperArm",
    "rightUpperArm",
    "leftUpperLeg",
    "rightUpperLeg",
}

REQUIRED_VISEME_CLIPS = {"A", "I", "U", "E", "O"}
REQUIRED_FACE_CLIPS = {"Blink", "Joy"}
EMOTION_CLIPS = {"Joy", "Angry", "Sorrow", "Fun", "Surprised"}
ARKIT_CLIP_COUNT = 52


@dataclass(frozen=True)
class VtuberQualityIssue:
    """One quality finding."""

    code: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class VtuberQualityReport:
    """Result of the human-facing VTuber quality gate."""

    passed: bool
    score: float
    issues: list[VtuberQualityIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[VtuberQualityIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[VtuberQualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        msg = "; ".join(issue.message for issue in self.errors) or "VTuber quality gate failed"
        raise ValueError(msg)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            data = obj.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass
    return getattr(obj, "__dict__", {}) or {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _vrm_extension(raw_gltf: Any) -> dict:
    extensions = _as_dict(_get(raw_gltf, "extensions", {}))
    return _as_dict(extensions.get("VRM") or extensions.get("VRMC_vrm"))


def _extract_human_bones(vrm_ext: dict) -> set[str]:
    humanoid = _as_dict(vrm_ext.get("humanoid"))
    human_bones = humanoid.get("humanBones", [])
    out: set[str] = set()
    for bone in human_bones or []:
        data = _as_dict(bone)
        name = data.get("bone")
        if isinstance(name, str):
            out.add(name)
    return out


def _extract_blendshape_names(vrm_ext: dict) -> set[str]:
    blend_master = _as_dict(vrm_ext.get("blendShapeMaster"))
    groups = blend_master.get("blendShapeGroups", [])
    names: set[str] = set()
    for group in groups or []:
        data = _as_dict(group)
        name = data.get("name") or data.get("presetName")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _mesh_stats(raw_gltf: Any) -> dict[str, int]:
    meshes = _get(raw_gltf, "meshes", []) or []
    skins = _get(raw_gltf, "skins", []) or []
    primitives = 0
    morph_targets = 0
    for mesh in meshes:
        for primitive in (_get(mesh, "primitives", []) or []):
            primitives += 1
            morph_targets += len(_get(primitive, "targets", []) or [])
    return {
        "mesh_count": len(meshes),
        "skin_count": len(skins),
        "primitive_count": primitives,
        "morph_target_count": morph_targets,
    }


def _score(errors: int, warnings: int) -> float:
    return max(0.0, 10.0 - errors * 2.5 - warnings * 0.5)


class VtuberQualityGate:
    """Validate whether a VRM output is fit for a VTuber workflow."""

    def validate_path(self, vrm_path: Path | str, atlas_map: "AtlasMap | None" = None) -> VtuberQualityReport:
        from ..vrm.vrm_io import VRMFile

        vrm = VRMFile.load(vrm_path)
        return self.validate_vrm(vrm, atlas_map=atlas_map)

    def validate_vrm(self, vrm: Any, atlas_map: "AtlasMap | None" = None) -> VtuberQualityReport:
        raw = _get(vrm, "raw", vrm)
        vrm_ext = _vrm_extension(raw)
        issues: list[VtuberQualityIssue] = []
        details: dict[str, Any] = {}

        if not vrm_ext:
            issues.append(VtuberQualityIssue("missing_vrm_extension", "缺少 VRM extension，不是可用 VTuber VRM"))
            return VtuberQualityReport(False, _score(1, 0), issues, details)

        bones = _extract_human_bones(vrm_ext)
        missing_bones = sorted(REQUIRED_HUMANOID_BONES - bones)
        details["human_bone_count"] = len(bones)
        details["missing_required_bones"] = missing_bones
        if missing_bones:
            issues.append(VtuberQualityIssue(
                "missing_humanoid_bones",
                "缺少 humanoid 骨架：" + ", ".join(missing_bones),
            ))

        blendshapes = _extract_blendshape_names(vrm_ext)
        details["blendshape_count"] = len(blendshapes)
        missing_visemes = sorted(REQUIRED_VISEME_CLIPS - blendshapes)
        if missing_visemes:
            issues.append(VtuberQualityIssue(
                "missing_visemes",
                "缺少口型 blendshape：" + ", ".join(missing_visemes),
            ))
        if not (REQUIRED_FACE_CLIPS & blendshapes):
            issues.append(VtuberQualityIssue("missing_blink_or_expression", "缺少眨眼或基本表情 blendshape"))

        emotion_count = len(EMOTION_CLIPS & blendshapes)
        details["emotion_clip_count"] = emotion_count
        if emotion_count < 2:
            issues.append(VtuberQualityIssue(
                "weak_expression_set",
                "表情組太少，角色看起來不像可直播的 VTuber",
                severity="warning",
            ))

        arkit_count = sum(1 for name in blendshapes if name.startswith(("eye", "jaw", "mouth", "brow", "cheek", "nose", "tongue")))
        details["arkit_like_clip_count"] = arkit_count
        if arkit_count < ARKIT_CLIP_COUNT:
            issues.append(VtuberQualityIssue(
                "missing_perfect_sync",
                f"Perfect Sync/ARKit clip 不完整：{arkit_count}/{ARKIT_CLIP_COUNT}",
                severity="warning",
            ))

        mesh_stats = _mesh_stats(raw)
        details.update(mesh_stats)
        if mesh_stats["mesh_count"] < 1 or mesh_stats["primitive_count"] < 1:
            issues.append(VtuberQualityIssue("missing_mesh", "缺少可見 mesh"))
        if mesh_stats["skin_count"] < 1:
            issues.append(VtuberQualityIssue("missing_skin", "缺少 skin/skinned mesh，無法像 VTuber 一樣動作"))

        if atlas_map is not None:
            image_count = len(_get(raw, "images", []) or [])
            required_indices = {
                "face_skin": atlas_map.face_skin_index,
                "hair": atlas_map.hair_index,
                "eye_iris": atlas_map.eye_iris_index,
            }
            details["image_count"] = image_count
            for label, index in required_indices.items():
                if index < 0 or index >= image_count:
                    issues.append(VtuberQualityIssue(
                        f"missing_{label}_texture",
                        f"缺少 {label} 貼圖 index={index}",
                    ))
                    continue
                if hasattr(vrm, "get_image_pil"):
                    try:
                        image = vrm.get_image_pil(index)
                        width, height = image.size
                        if width < 64 or height < 64:
                            issues.append(VtuberQualityIssue(
                                f"tiny_{label}_texture",
                                f"{label} 貼圖尺寸太小：{width}x{height}",
                            ))
                    except Exception as e:  # noqa: BLE001
                        issues.append(VtuberQualityIssue(
                            f"broken_{label}_texture",
                            f"{label} 貼圖無法讀取：{type(e).__name__}",
                        ))

        error_count = sum(issue.severity == "error" for issue in issues)
        warning_count = sum(issue.severity == "warning" for issue in issues)
        report = VtuberQualityReport(
            passed=error_count == 0,
            score=_score(error_count, warning_count),
            issues=issues,
            details=details,
        )
        if report.passed:
            _log.info("VTuber quality gate passed: score={:.1f} details={}", report.score, report.details)
        else:
            _log.warning("VTuber quality gate failed: score={:.1f} issues={}", report.score, report.issues)
        return report
