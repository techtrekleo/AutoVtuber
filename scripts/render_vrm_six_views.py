"""Render a VRM/GLB into six orthographic preview PNGs.

This renderer intentionally avoids OpenGL so it works in constrained desktop
test environments. It reads mesh geometry with pygltflib, applies node
transforms, and draws painter-sorted shaded triangles with matplotlib.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pygltflib import GLTF2


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _mat_from_trs(node) -> np.ndarray:
    if node.matrix:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T

    t = np.array(node.translation or [0, 0, 0], dtype=np.float64)
    s = np.array(node.scale or [1, 1, 1], dtype=np.float64)
    x, y, z, w = node.rotation or [0, 0, 0, 1]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r[:3, :3] @ np.diag(s)
    m[:3, 3] = t
    return m


def _walk_nodes(gltf: GLTF2) -> list[tuple[int, np.ndarray]]:
    scene_index = gltf.scene if gltf.scene is not None else 0
    roots = gltf.scenes[scene_index].nodes or []
    out: list[tuple[int, np.ndarray]] = []

    def walk(node_index: int, parent: np.ndarray) -> None:
        node = gltf.nodes[node_index]
        world = parent @ _mat_from_trs(node)
        if node.mesh is not None:
            out.append((node.mesh, world))
        for child in node.children or []:
            walk(child, world)

    for root in roots:
        walk(root, np.eye(4, dtype=np.float64))
    return out


def _read_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    dtype = COMPONENT_DTYPES[accessor.componentType]
    item_count = TYPE_COUNTS[accessor.type]
    byte_offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = view.byteStride or (np.dtype(dtype).itemsize * item_count)
    total_bytes = (accessor.count - 1) * byte_stride + np.dtype(dtype).itemsize * item_count
    raw = memoryview(blob)[byte_offset: byte_offset + total_bytes]

    if byte_stride == np.dtype(dtype).itemsize * item_count:
        arr = np.frombuffer(raw, dtype=dtype, count=accessor.count * item_count)
        return arr.reshape(accessor.count, item_count).copy()

    rows = []
    item_bytes = np.dtype(dtype).itemsize * item_count
    for i in range(accessor.count):
        start = i * byte_stride
        rows.append(np.frombuffer(raw[start: start + item_bytes], dtype=dtype, count=item_count))
    return np.vstack(rows).copy()


def _material_color(gltf: GLTF2, material_index: int | None) -> tuple[float, float, float]:
    if material_index is None or material_index >= len(gltf.materials or []):
        return (0.82, 0.72, 0.64)
    material = gltf.materials[material_index]
    pbr = material.pbrMetallicRoughness
    if pbr and pbr.baseColorFactor:
        r, g, b, _a = pbr.baseColorFactor
        return (float(r), float(g), float(b))
    return (0.82, 0.72, 0.64)


def load_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    gltf = GLTF2.load_binary(str(path))
    triangles: list[np.ndarray] = []
    colors: list[tuple[float, float, float]] = []
    for mesh_index, world in _walk_nodes(gltf):
        mesh = gltf.meshes[mesh_index]
        for primitive in mesh.primitives:
            if primitive.attributes.POSITION is None:
                continue
            vertices = _read_accessor(gltf, primitive.attributes.POSITION).astype(np.float64)
            vertices_h = np.c_[vertices, np.ones(len(vertices))]
            vertices = (vertices_h @ world.T)[:, :3]
            if primitive.indices is not None:
                indices = _read_accessor(gltf, primitive.indices).reshape(-1).astype(np.int64)
            else:
                indices = np.arange(len(vertices), dtype=np.int64)
            if len(indices) < 3:
                continue
            tris = vertices[indices[: len(indices) // 3 * 3].reshape(-1, 3)]
            triangles.extend(tris)
            colors.extend([_material_color(gltf, primitive.material)] * len(tris))
    if not triangles:
        raise RuntimeError("No triangles found")
    return np.stack(triangles), np.array(colors, dtype=np.float64)


def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = direction / (np.linalg.norm(direction) + 1e-9)
    up_guess = np.array([0, 1, 0], dtype=np.float64)
    if abs(np.dot(forward, up_guess)) > 0.92:
        up_guess = np.array([0, 0, 1], dtype=np.float64)
    right = np.cross(up_guess, forward)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(forward, right)
    up /= np.linalg.norm(up) + 1e-9
    return right, up, forward


def render_view(
    triangles: np.ndarray,
    colors: np.ndarray,
    direction: np.ndarray,
    label: str,
    out_path: Path,
    size: int = 960,
) -> None:
    right, up, forward = _basis(direction)
    points = triangles.reshape(-1, 3)
    center = points.mean(axis=0)
    centered = triangles - center
    x = np.einsum("...j,j->...", centered, right)
    y = np.einsum("...j,j->...", centered, up)
    z = np.einsum("...j,j->...", centered, forward)

    tri_normals = np.cross(centered[:, 1] - centered[:, 0], centered[:, 2] - centered[:, 0])
    tri_normals /= np.linalg.norm(tri_normals, axis=1, keepdims=True) + 1e-9
    facing = np.einsum("ij,j->i", tri_normals, forward)
    shade = np.clip(0.38 + 0.62 * np.abs(facing), 0.25, 1.0)
    face_colors = np.clip(colors * shade[:, None], 0, 1)

    depth = z.mean(axis=1)
    order = np.argsort(depth)
    ordered_colors = face_colors[order]

    span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-6) * 0.62
    canvas = Image.new("RGB", (size, size), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    def to_pixel(px: np.ndarray, py: np.ndarray) -> list[tuple[float, float]]:
        sx = (px / (2 * span) + 0.5) * size
        sy = (0.5 - py / (2 * span)) * size
        return list(zip(sx.tolist(), sy.tolist()))

    for rank, tri_index in enumerate(order):
        poly = to_pixel(x[tri_index], y[tri_index])
        color = tuple(np.clip(ordered_colors[rank] * 255, 0, 255).astype(np.uint8).tolist())
        draw.polygon(poly, fill=color)

    draw.rectangle((12, 12, 108, 42), fill=(0, 0, 0))
    draw.text((22, 21), label, fill=(245, 245, 245))
    canvas.save(out_path)


def make_contact_sheet(paths: list[Path], labels: list[str], out_path: Path) -> None:
    thumbs = [Image.open(p).convert("RGB").resize((420, 420), Image.LANCZOS) for p in paths]
    canvas = Image.new("RGB", (3 * 420, 2 * 456), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        x = (i % 3) * 420
        y = (i // 3) * 456
        canvas.paste(thumb, (x, y + 36))
        draw.text((x + 14, y + 10), label, fill=(240, 240, 240))
    canvas.save(out_path)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: render_vrm_six_views.py path/to/model.vrm")
        return 2
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    triangles, colors = load_triangles(src)
    specs = [
        # VRoid avatars face -Z in this base sample.
        ("front", np.array([0.0, 0.0, -1.0])),
        ("back", np.array([0.0, 0.0, 1.0])),
        ("left", np.array([-1.0, 0.0, 0.0])),
        ("right", np.array([1.0, 0.0, 0.0])),
        ("top", np.array([0.0, 1.0, 0.0])),
        ("bottom", np.array([0.0, -1.0, 0.0])),
    ]
    outputs = []
    labels = []
    for name, direction in specs:
        out = out_dir / f"{src.stem}_{name}.png"
        render_view(triangles, colors, direction, name, out)
        outputs.append(out)
        labels.append(name)
        print(out)
    sheet = out_dir / f"{src.stem}_six_views.png"
    make_contact_sheet(outputs, labels, sheet)
    print(sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
