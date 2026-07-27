#!/usr/bin/env python3
"""Generate self-contained glTFs for the Box3D physics PoC.

Each mesh is emitted as its own .gltf next to this script, with an embedded
base64 buffer (no external .bin), a single opaque PBR material, and per-vertex
normals. Meshes are authored around half-extent ~1 so Box3D collider dims map
cleanly from mesh bounds.

Outputs: box3d_cube.gltf, box3d_ground.gltf, box3d_sphere.gltf,
box3d_capsule.gltf, box3d_hull.gltf.
"""
import struct, base64, json, os, math

ASSETS = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# glTF emit
# ---------------------------------------------------------------------------
def make_gltf(name, color, positions, normals, indices):
    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_bytes = b"".join(struct.pack("<I", i) for i in indices)
    buf = pos_bytes + nrm_bytes + idx_bytes
    pos_off, nrm_off, idx_off = 0, len(pos_bytes), len(pos_bytes) + len(nrm_bytes)

    mn = [min(p[i] for p in positions) for i in range(3)]
    mx = [max(p[i] for p in positions) for i in range(3)]

    return {
        "asset": {"version": "2.0", "generator": "box3d-demo asset generator"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": 2,
                "material": 0,
            }],
        }],
        "materials": [{
            "name": name + "_mat",
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "alphaMode": "OPAQUE",
        }],
        "buffers": [{
            "byteLength": len(buf),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode(),
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }


# ---------------------------------------------------------------------------
# geometry builders (all centered at origin)
# ---------------------------------------------------------------------------
def build_box(hx=1.0, hy=1.0, hz=1.0):
    faces = [
        ([1, 0, 0], [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ([-1, 0, 0], [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ([0, 1, 0], [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ([0, -1, 0], [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ([0, 0, 1], [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ([0, 0, -1], [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
    ]
    positions, normals, indices = [], [], []
    for n, corners in faces:
        base = len(positions)
        for c in corners:
            positions.append(c)
            normals.append(n)
        indices += [base + 0, base + 1, base + 2, base + 0, base + 2, base + 3]
    return positions, normals, indices


def build_sphere(radius=1.0, stacks=12, slices=16):
    positions, normals, indices = [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            positions.append((radius * nx, radius * ny, radius * nz))
            normals.append((nx, ny, nz))
    row = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * row + j
            b = a + row
            indices += [a, b, a + 1, a + 1, b, b + 1]
    return positions, normals, indices


def build_capsule(radius=0.5, half=1.0, slices=16, stacks=6):
    """Capsule along Y: cylinder from y=-half..half capped by hemispheres of `radius`.
    Bounds -> extents (radius, half+radius, radius)."""
    positions, normals, indices = [], [], []
    rings = []
    # top hemisphere phi 0..pi/2 (pole at y=half+radius down to equator y=half)
    for i in range(stacks + 1):
        phi = (math.pi / 2) * i / stacks
        ring = []
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            positions.append((radius * nx, half + radius * ny, radius * nz))
            normals.append((nx, ny, nz))
            ring.append(len(positions) - 1)
        rings.append(ring)
    # bottom hemisphere phi pi/2..pi (equator y=-half down to pole y=-half-radius)
    for i in range(stacks + 1):
        phi = math.pi / 2 + (math.pi / 2) * i / stacks
        ring = []
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            positions.append((radius * nx, -half + radius * ny, radius * nz))
            normals.append((nx, ny, nz))
            ring.append(len(positions) - 1)
        rings.append(ring)
    for i in range(len(rings) - 1):
        for j in range(slices):
            a, b = rings[i][j], rings[i + 1][j]
            a1, b1 = rings[i][j + 1], rings[i + 1][j + 1]
            indices += [a, b, a1, a1, b, b1]
    return positions, normals, indices


def build_icosahedron(radius=1.0):
    """Low-poly convex 'rock' (12 verts, 20 faces) for the hull collider."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
           (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
           (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    positions, normals = [], []
    for v in raw:
        L = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        n = (v[0] / L, v[1] / L, v[2] / L)
        positions.append((radius * n[0], radius * n[1], radius * n[2]))
        normals.append(n)
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    indices = []
    for f in faces:
        indices += [f[0], f[1], f[2]]
    return positions, normals, indices


# ---------------------------------------------------------------------------
targets = [
    ("box3d_cube",    [0.90, 0.45, 0.12, 1.0], build_box(1.0, 1.0, 1.0)),   # orange
    ("box3d_ground",  [0.20, 0.45, 0.55, 1.0], build_box(1.0, 1.0, 1.0)),   # teal (scaled in scene)
    ("box3d_sphere",  [0.30, 0.70, 0.35, 1.0], build_sphere(1.0)),          # green
    ("box3d_capsule", [0.65, 0.35, 0.75, 1.0], build_capsule(0.5, 1.0)),    # purple
    ("box3d_hull",    [0.85, 0.75, 0.20, 1.0], build_icosahedron(1.0)),     # yellow
]
for name, color, (positions, normals, indices) in targets:
    out = os.path.join(ASSETS, name + ".gltf")
    with open(out, "w") as f:
        json.dump(make_gltf(name, color, positions, normals, indices), f, indent=2)
    print(f"wrote {out}  verts={len(positions)} tris={len(indices)//3} color={color}")
