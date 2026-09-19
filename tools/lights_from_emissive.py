#!/usr/bin/env python3
"""
Generate punctual lights directly from Bistro_new's own emissive geometry.

The alternative, tools/transplant_lights.py, copies zeux's 97 lights across and corrects for the
origin difference between the two exports. That alignment is correct (every light lands within a
fraction of a unit of real geometry) but it cannot fix a content mismatch: Bistro_new carries 73
string-light bulbs while zeux only had 57 coloured lights, and the two exports differ by 1377
primitives overall. The result is bulbs that glow with no light on them and lights sitting where a
fixture used to be.

Generating from the target scene's own emissive meshes removes the problem by construction -- there
is nothing to align, because each light is placed at the centroid of the bulb it belongs to and
takes that material's own colour.

Intensity and range are still calibrated from zeux, which is where the night look was tuned.

Reads  : assets/Bistro_new.glb, assets/niagara_bistro/bistro.gltf (calibration + the sun)
Writes : assets/bistro_glb/bistro.glb        (both inputs left untouched)
"""
import json, struct, math, os, sys
from collections import defaultdict

# Emissive materials that represent an actual light source. Glass covers (Spotlight_Glass,
# MASTER_Focus_Glass) are deliberately excluded: they sit on top of Spotlight_Emissive and would
# double every fixture. Signage (Bistro_Sign_Letters, Shopsign_Pharmacy) is emissive but lights
# nothing, so it stays a glowing surface rather than becoming a punctual light.
LIGHT_MATERIALS = ("StringLights", "Lantern", "Spotlight_Emissive")

# Bulbs modelled as several primitives would otherwise each get their own light.
MERGE_DIST = 0.4

# Max extent of a single bulb. Vertices closer than this are treated as one bulb; the gap between
# neighbouring bulbs on a strand is far larger, so the split is unambiguous.
BULB_RADIUS = 0.5


def mat_identity():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def mat_mul(a, b):
    out = [0.0] * 16
    for c in range(4):
        for r in range(4):
            out[c * 4 + r] = sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4))
    return out


def node_local_mat(n):
    if 'matrix' in n:
        return list(n['matrix'])
    t = n.get('translation', [0, 0, 0])
    x, y, z, w = n.get('rotation', [0, 0, 0, 1])
    s = n.get('scale', [1, 1, 1])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [(1 - 2 * (yy + zz)) * s[0], (2 * (xy + wz)) * s[0], (2 * (xz - wy)) * s[0], 0,
            (2 * (xy - wz)) * s[1], (1 - 2 * (xx + zz)) * s[1], (2 * (yz + wx)) * s[1], 0,
            (2 * (xz + wy)) * s[2], (2 * (yz - wx)) * s[2], (1 - 2 * (xx + yy)) * s[2], 0,
            t[0], t[1], t[2], 1]


def xform(m, p):
    return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]]


def read_accessor_vec3(d, binchunk, index):
    """Vertex positions for one accessor. Needed because a primitive can hold a whole strand of
    bulbs, and its bounding-box centre then sits between them rather than on any one."""
    a = d['accessors'][index]
    if a.get('componentType') != 5126 or a.get('type') != 'VEC3':
        return []
    bv = d['bufferViews'][a['bufferView']]
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = bv.get('byteStride') or 12
    out = []
    for i in range(a['count']):
        o = base + i * stride
        out.append(struct.unpack_from('<3f', binchunk, o))
    return out


def cluster(points, radius):
    """Greedy spatial clustering. Bulbs are far apart relative to their size, so this is enough."""
    centres = []
    counts = []
    r2 = radius * radius
    for p in points:
        for j, c in enumerate(centres):
            if (p[0]-c[0])**2 + (p[1]-c[1])**2 + (p[2]-c[2])**2 < r2:
                n = counts[j]
                centres[j] = [(c[k]*n + p[k])/(n+1) for k in range(3)]
                counts[j] = n + 1
                break
        else:
            centres.append(list(p))
            counts.append(1)
    return centres


def read_glb(path):
    with open(path, 'rb') as f:
        magic, ver, total = struct.unpack('<4sII', f.read(12))
        if magic != b'glTF':
            sys.exit(f"{path}: not a GLB")
        js, binchunk = None, b''
        while f.tell() < total:
            clen, ctype = struct.unpack('<I4s', f.read(8))
            data = f.read(clen)
            if ctype == b'JSON':
                js = json.loads(data.decode('utf-8'))
            elif ctype == b'BIN\x00':
                binchunk = data
        return js, binchunk


def write_glb(path, js, binchunk):
    jsdata = json.dumps(js, separators=(',', ':')).encode('utf-8')
    jsdata += b' ' * ((4 - len(jsdata) % 4) % 4)
    binpad = b'\x00' * ((4 - len(binchunk) % 4) % 4)
    total = 12 + 8 + len(jsdata) + (8 + len(binchunk) + len(binpad) if binchunk else 0)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I4s', len(jsdata), b'JSON'))
        f.write(jsdata)
        if binchunk:
            f.write(struct.pack('<I4s', len(binchunk) + len(binpad), b'BIN\x00'))
            f.write(binchunk)
            f.write(binpad)


def calibrate(src_path):
    """Median intensity and range from zeux, split by colour. That scene is where night was tuned."""
    src = json.load(open(src_path))
    lights = src.get('extensions', {}).get('KHR_lights_punctual', {}).get('lights', [])
    groups = defaultdict(lambda: ([], []))
    sun = None
    for l in lights:
        if l.get('type') == 'directional':
            sun = sun or l
            continue
        c = l.get('color', [1, 1, 1])
        key = 'white' if min(c) > 0.85 else 'colour'
        groups[key][0].append(l.get('intensity', 1.0))
        groups[key][1].append(l.get('range', 5.0))
    out = {}
    for k, (ints, rngs) in groups.items():
        ints.sort(); rngs.sort()
        out[k] = (ints[len(ints) // 2], rngs[len(rngs) // 2])
    return out, sun


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src_path = os.path.join(root, 'assets/niagara_bistro/bistro.gltf')
    dst_path = os.path.join(root, 'assets/Bistro_new.glb')
    out_path = os.path.join(root, 'assets/bistro_glb/bistro.glb')

    cal, sun = calibrate(src_path)
    print("calibration from zeux (median intensity, range):")
    for k, v in sorted(cal.items()):
        print(f"   {k:6s}: intensity {v[0]:8.2f}   range {v[1]:5.2f}")

    d, binchunk = read_glb(dst_path)
    mats = d.get('materials', [])
    nodes = d.get('nodes', [])

    wanted = {i: m.get('name', '') for i, m in enumerate(mats)
              if any(k in m.get('name', '') for k in LIGHT_MATERIALS)}
    if not wanted:
        sys.exit("no emissive light materials matched; check LIGHT_MATERIALS")

    found = []  # (centroid, material index)

    def walk(i, parent):
        n = nodes[i]
        m = mat_mul(parent, node_local_mat(n))
        mi = n.get('mesh')
        if mi is not None:
            for prim in d['meshes'][mi].get('primitives', []):
                if prim.get('material') not in wanted:
                    continue
                ai = prim.get('attributes', {}).get('POSITION')
                if ai is None:
                    continue
                verts = read_accessor_vec3(d, binchunk, ai)
                if not verts:
                    continue
                world = [xform(m, v) for v in verts]
                for c in cluster(world, BULB_RADIUS):
                    found.append((c, prim['material']))
        for c in n.get('children', []):
            walk(c, m)

    for r in d['scenes'][0]['nodes']:
        walk(r, mat_identity())

    # Merge primitives belonging to the same physical bulb.
    merged = []
    for pos, mi in found:
        for j, (p2, mi2, n2) in enumerate(merged):
            if mi2 == mi and sum((pos[k] - p2[k]) ** 2 for k in range(3)) < MERGE_DIST ** 2:
                merged[j] = ([(p2[k] * n2 + pos[k]) / (n2 + 1) for k in range(3)], mi2, n2 + 1)
                break
        else:
            merged.append((pos, mi, 1))

    lights, light_nodes = [], []
    per_mat = defaultdict(int)

    for pos, mi, _ in merged:
        factor = mats[mi].get('emissiveFactor', [1, 1, 1])
        peak = max(factor) or 1.0
        color = [min(1.0, v / peak) for v in factor]
        if max(color) <= 0.0:
            color = [1.0, 1.0, 1.0]  # texture-driven emissive (Lantern) carries no useful factor

        # Keyed on the material, NOT the colour: a third of the string bulbs are white, and
        # classifying those by colour would hand them the street-lamp calibration -- four times
        # the intensity and twice the reach of the bulb they belong to.
        key = 'colour' if 'StringLights' in mats[mi].get('name', '') else 'white'
        intensity, rng = cal.get(key, cal.get('colour', (100.0, 5.0)))

        lights.append({'type': 'point', 'color': [round(v, 4) for v in color],
                       'intensity': round(intensity, 3), 'range': round(rng, 3),
                       'name': mats[mi].get('name', 'emissive')})
        light_nodes.append({'name': f"light_{mats[mi].get('name','emissive')}_{len(light_nodes)}",
                            'translation': [round(v, 5) for v in pos],
                            'extensions': {'KHR_lights_punctual': {'light': len(lights) - 1}}})
        per_mat[mats[mi].get('name', '?')] += 1

    # Bistro_new has no sun; take zeux's directional light as authored (direction is a rotation,
    # so the origin difference between the two scenes does not affect it).
    if sun is not None:
        src = json.load(open(src_path))
        for n in src['nodes']:
            e = n.get('extensions', {}).get('KHR_lights_punctual')
            if e is not None and src['extensions']['KHR_lights_punctual']['lights'][e['light']].get('type') == 'directional':
                lights.append(dict(sun))
                light_nodes.append({'name': 'Sun', 'rotation': n.get('rotation', [0, 0, 0, 1]),
                                    'translation': [0, 0, 0],
                                    'extensions': {'KHR_lights_punctual': {'light': len(lights) - 1}}})
                break

    d.setdefault('extensions', {})['KHR_lights_punctual'] = {'lights': lights}
    base = len(nodes)
    nodes.extend(light_nodes)
    d['nodes'] = nodes
    d['scenes'][0].setdefault('nodes', []).extend(range(base, base + len(light_nodes)))
    used = d.setdefault('extensionsUsed', [])
    if 'KHR_lights_punctual' not in used:
        used.append('KHR_lights_punctual')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_glb(out_path, d, binchunk)

    print(f"\nemissive primitives matched : {len(found)}")
    print(f"after merging within {MERGE_DIST}     : {len(merged)}")
    print("lights per source material:")
    for k, v in sorted(per_mat.items(), key=lambda x: -x[1]):
        print(f"   {v:4d}  {k}")
    print(f"\ntotal lights: {len(lights)} (incl. sun: {sun is not None})")
    ys = [n['translation'][1] for n in light_nodes]
    print(f"height range: {min(ys):.2f} .. {max(ys):.2f}")
    print(f"\nwrote {out_path}  ({os.path.getsize(out_path)/1048576:.0f} MB)")


main()
