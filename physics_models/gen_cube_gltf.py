#!/usr/bin/env python3
"""Generate self-contained unit cube glTFs (half-extent 1) for the box3d demo.
24 verts (per-face normals), 36 indices, one opaque PBR material each.
Embedded base64 buffer -> no external .bin.
Outputs (next to this script): box3d_cube.gltf (orange body) and box3d_ground.gltf (teal).
"""
import struct, base64, json, os

# (normal, [4 outward CCW corners]) per face, cube spans [-1, 1]
faces = [
    ([ 1, 0, 0], [(1,-1, 1),(1,-1,-1),(1, 1,-1),(1, 1, 1)]),  # +X
    ([-1, 0, 0], [(-1,-1,-1),(-1,-1, 1),(-1, 1, 1),(-1, 1,-1)]),  # -X
    ([ 0, 1, 0], [(-1, 1, 1),(1, 1, 1),(1, 1,-1),(-1, 1,-1)]),  # +Y
    ([ 0,-1, 0], [(-1,-1,-1),(1,-1,-1),(1,-1, 1),(-1,-1, 1)]),  # -Y
    ([ 0, 0, 1], [(-1,-1, 1),(1,-1, 1),(1, 1, 1),(-1, 1, 1)]),  # +Z
    ([ 0, 0,-1], [(1,-1,-1),(-1,-1,-1),(-1, 1,-1),(1, 1,-1)]),  # -Z
]

positions, normals, indices = [], [], []
for n, corners in faces:
    base = len(positions)
    for c in corners:
        positions.append(c)
        normals.append(n)
    indices += [base+0, base+1, base+2, base+0, base+2, base+3]

pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
idx_bytes = b"".join(struct.pack("<I", i) for i in indices)

# pad each section start to 4 bytes (already 4-aligned here)
buf = pos_bytes + nrm_bytes + idx_bytes
pos_off, nrm_off, idx_off = 0, len(pos_bytes), len(pos_bytes) + len(nrm_bytes)

def make_gltf(name, color):
    return {
        "asset": {"version": "2.0", "generator": "box3d-demo cube generator"},
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
            {"bufferView": 0, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": [-1,-1,-1], "max": [1,1,1]},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }

# Write next to this script (the physics_models/ folder).
ASSETS = os.path.dirname(os.path.abspath(__file__))
targets = [
    ("box3d_cube",   [0.90, 0.45, 0.12, 1.0]),  # warm orange body
    ("box3d_ground", [0.20, 0.45, 0.55, 1.0]),  # cool teal/blue-gray slab
]
for name, color in targets:
    out = os.path.join(ASSETS, name + ".gltf")
    with open(out, "w") as f:
        json.dump(make_gltf(name, color), f, indent=2)
    print("wrote", out, "color", color)
print("buffer bytes:", len(buf), "verts:", len(positions), "indices:", len(indices))
