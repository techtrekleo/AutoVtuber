from __future__ import annotations

from dataclasses import dataclass, field

from autovtuber.pipeline.vtuber_quality import VtuberQualityGate


@dataclass
class DummyPrimitive:
    targets: list[dict] = field(default_factory=lambda: [{}])


@dataclass
class DummyMesh:
    primitives: list[DummyPrimitive] = field(default_factory=lambda: [DummyPrimitive()])


@dataclass
class DummyRaw:
    extensions: dict
    meshes: list[DummyMesh] = field(default_factory=lambda: [DummyMesh()])
    skins: list[dict] = field(default_factory=lambda: [{}])
    images: list[dict] = field(default_factory=list)


@dataclass
class DummyVRM:
    raw: DummyRaw


def make_vrm(
    *,
    bones: set[str] | None = None,
    blendshapes: set[str] | None = None,
    meshes: list[DummyMesh] | None = None,
    skins: list[dict] | None = None,
) -> DummyVRM:
    bones = bones or {
        "hips",
        "spine",
        "head",
        "neck",
        "leftUpperArm",
        "rightUpperArm",
        "leftUpperLeg",
        "rightUpperLeg",
    }
    blendshapes = blendshapes or {
        "A",
        "I",
        "U",
        "E",
        "O",
        "Blink",
        "Joy",
        "Angry",
        "Sorrow",
        "Fun",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "jawOpen",
        "mouthSmileLeft",
        "mouthSmileRight",
        "browDownLeft",
    }
    return DummyVRM(
        raw=DummyRaw(
            extensions={
                "VRM": {
                    "humanoid": {
                        "humanBones": [{"bone": bone, "node": i} for i, bone in enumerate(sorted(bones))]
                    },
                    "blendShapeMaster": {
                        "blendShapeGroups": [{"name": name, "binds": []} for name in sorted(blendshapes)]
                    },
                }
            },
            meshes=[DummyMesh()] if meshes is None else meshes,
            skins=[{}] if skins is None else skins,
        )
    )


def test_quality_gate_passes_structural_vtuber():
    report = VtuberQualityGate().validate_vrm(make_vrm())
    assert report.passed
    assert report.score > 0
    assert report.details["human_bone_count"] >= 8
    assert report.details["blendshape_count"] >= 10


def test_quality_gate_rejects_missing_humanoid_bones():
    report = VtuberQualityGate().validate_vrm(make_vrm(bones={"head"}))
    assert not report.passed
    assert any(issue.code == "missing_humanoid_bones" for issue in report.errors)


def test_quality_gate_rejects_missing_mouth_shapes():
    report = VtuberQualityGate().validate_vrm(make_vrm(blendshapes={"Blink", "Joy"}))
    assert not report.passed
    assert any(issue.code == "missing_visemes" for issue in report.errors)


def test_quality_gate_rejects_static_mesh_without_skin():
    report = VtuberQualityGate().validate_vrm(make_vrm(skins=[]))
    assert not report.passed
    assert any(issue.code == "missing_skin" for issue in report.errors)


def test_quality_gate_rejects_non_vrm():
    report = VtuberQualityGate().validate_vrm(DummyVRM(raw=DummyRaw(extensions={})))
    assert not report.passed
    assert any(issue.code == "missing_vrm_extension" for issue in report.errors)
