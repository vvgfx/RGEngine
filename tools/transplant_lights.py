#!/usr/bin/env python3
"""
Copy the 97 KHR_lights_punctual lights out of zeux's Bistro glTF and into Bistro_new.glb.

Bistro_new is fully metallic-roughness but ships no lights; zeux's scene has the lights but is
spec-gloss. The two share a coordinate system (verified: zeux's light positions fall inside
Bistro_new's world bounds at matching scale and Y-up orientation), so the lights transfer directly.

The two scenes do NOT share an origin. Bistro_new is a separate Blender re-export, so although
its vertical datum matches zeux exactly (y bounds -4.73..27.21 in both, scale 1.000), it is
translated in the horizontal plane.

The correction was measured by rasterising both scenes' geometry into an XZ occupancy grid and
searching offsets by brute force -- deliberately NOT by FFT cross-correlation, whose peak-index
sign convention is easy to get backwards (it was, first time round, and produced worse alignment
than no correction at all). Grid overlap: 232 uncorrected, 96 with the sign flipped, 938 here.
The peak is sharp and symmetric, neighbours at +/-0.25 units scoring 842-885.

Y needs no correction: bounds are -4.73..27.21 in both scenes, scale 1.000. Testing 0/90/180/270
degree Y-rotations found no rotation either.

Both inputs are read-only; the result is written to a new file.
"""

# Measured, not guessed -- see the module docstring.
ALIGN_OFFSET = (11.0, 0.0, -12.0)
import json, struct, sys, math, os

def mat_identity():
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

def mat_mul(a, b):
    # column-major 4x4, matching glTF
    out = [0.0]*16
    for c in range(4):
        for r in range(4):
            out[c*4+r] = sum(a[k*4+r] * b[c*4+k] for k in range(4))
    return out

def trs_to_mat(t, r, s):
    x,y,z,w = r
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    m = [
        (1-2*(yy+zz))*s[0], (2*(xy+wz))*s[0],   (2*(xz-wy))*s[0],   0,
        (2*(xy-wz))*s[1],   (1-2*(xx+zz))*s[1], (2*(yz+wx))*s[1],   0,
        (2*(xz+wy))*s[2],   (2*(yz-wx))*s[2],   (1-2*(xx+yy))*s[2], 0,
        t[0], t[1], t[2], 1,
    ]
    return m

def node_local_mat(n):
    if 'matrix' in n:
        return list(n['matrix'])
    return trs_to_mat(n.get('translation',[0,0,0]),
                      n.get('rotation',[0,0,0,1]),
                      n.get('scale',[1,1,1]))

def mat_to_trs(m):
    t = [m[12], m[13], m[14]]
    cols = [[m[0],m[1],m[2]], [m[4],m[5],m[6]], [m[8],m[9],m[10]]]
    s = [math.sqrt(sum(v*v for v in c)) or 1.0 for c in cols]
    r = [[cols[c][i]/s[c] for i in range(3)] for c in range(3)]
    # rotation matrix -> quaternion (column-major: r[col][row])
    m00,m10,m20 = r[0]; m01,m11,m21 = r[1]; m02,m12,m22 = r[2]
    tr = m00 + m11 + m22
    if tr > 0:
        k = math.sqrt(tr+1.0)*2
        q = [(m12-m21)/k, (m20-m02)/k, (m01-m10)/k, 0.25*k]
    elif m00 > m11 and m00 > m22:
        k = math.sqrt(1.0+m00-m11-m22)*2
        q = [0.25*k, (m10+m01)/k, (m20+m02)/k, (m12-m21)/k]
    elif m11 > m22:
        k = math.sqrt(1.0+m11-m00-m22)*2
        q = [(m10+m01)/k, 0.25*k, (m21+m12)/k, (m20-m02)/k]
    else:
        k = math.sqrt(1.0+m22-m00-m11)*2
        q = [(m20+m02)/k, (m21+m12)/k, 0.25*k, (m01-m10)/k]
    n = math.sqrt(sum(v*v for v in q)) or 1.0
    return t, [v/n for v in q], s

def read_glb(path):
    with open(path,'rb') as f:
        magic, ver, total = struct.unpack('<4sII', f.read(12))
        if magic != b'glTF':
            sys.exit(f"{path}: not a GLB")
        js = None; binchunk = b''
        while f.tell() < total:
            clen, ctype = struct.unpack('<I4s', f.read(8))
            data = f.read(clen)
            if ctype == b'JSON': js = json.loads(data.decode('utf-8'))
            elif ctype == b'BIN\x00': binchunk = data
        return js, binchunk

def write_glb(path, js, binchunk):
    jsdata = json.dumps(js, separators=(',',':')).encode('utf-8')
    jsdata += b' ' * ((4 - len(jsdata) % 4) % 4)          # chunks must be 4-byte aligned
    binpad = b'\x00' * ((4 - len(binchunk) % 4) % 4)
    total = 12 + 8 + len(jsdata) + (8 + len(binchunk) + len(binpad) if binchunk else 0)
    with open(path,'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I4s', len(jsdata), b'JSON')); f.write(jsdata)
        if binchunk:
            f.write(struct.pack('<I4s', len(binchunk)+len(binpad), b'BIN\x00'))
            f.write(binchunk); f.write(binpad)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src_path = os.path.join(root, 'assets/niagara_bistro/bistro.gltf')
    dst_path = os.path.join(root, 'assets/Bistro_new.glb')
    out_path = os.path.join(root, 'assets/bistro_glb/bistro.glb')

    src = json.load(open(src_path))
    dst, binchunk = read_glb(dst_path)

    src_lights = src.get('extensions',{}).get('KHR_lights_punctual',{}).get('lights',[])
    src_nodes = src['nodes']

    parent = {}
    for i,n in enumerate(src_nodes):
        for c in n.get('children',[]): parent[c] = i

    def world_mat(i):
        m = node_local_mat(src_nodes[i])
        p = parent.get(i)
        while p is not None:
            m = mat_mul(node_local_mat(src_nodes[p]), m)
            p = parent.get(p)
        return m

    dst_ext = dst.setdefault('extensions',{}).setdefault('KHR_lights_punctual',{}).setdefault('lights',[])
    light_base = len(dst_ext)
    dst_ext.extend(json.loads(json.dumps(src_lights)))      # deep copy

    dst_nodes = dst.setdefault('nodes',[])
    node_base = len(dst_nodes)
    added = []
    src_pos, out_pos = [], []

    for i,n in enumerate(src_nodes):
        li = n.get('extensions',{}).get('KHR_lights_punctual',{}).get('light')
        if li is None: continue
        t, r, s = mat_to_trs(world_mat(i))
        src_pos.append(list(t))
        t = [t[k] + ALIGN_OFFSET[k] for k in range(3)]
        out_pos.append(t)
        new = {'name': n.get('name', f'light_{i}'),
               'translation': [round(v,6) for v in t],
               'rotation': [round(v,6) for v in r],
               'extensions': {'KHR_lights_punctual': {'light': light_base + li}}}
        dst_nodes.append(new); added.append(node_base + len(added))

    dst.setdefault('scenes',[{}])[0].setdefault('nodes',[]).extend(added)
    used = dst.setdefault('extensionsUsed',[])
    if 'KHR_lights_punctual' not in used: used.append('KHR_lights_punctual')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_glb(out_path, dst, binchunk)

    types = {}
    for l in src_lights: types[l.get('type')] = types.get(l.get('type'),0)+1
    print(f"lights moved : {len(added)}  {types}")
    print(f"light defs   : {len(dst_ext)} (appended at index {light_base})")
    print(f"scene roots  : {len(dst['scenes'][0]['nodes'])}")
    print(f"alignment    : dx={ALIGN_OFFSET[0]:+.2f} dy={ALIGN_OFFSET[1]:+.2f} dz={ALIGN_OFFSET[2]:+.2f}")
    for lbl, ps in (("zeux world      ", src_pos), ("aligned to new  ", out_pos)):
        for k,ax in enumerate('xyz'):
            v=[p[k] for p in ps]
            print(f"  {lbl} {ax}: {min(v):8.2f} .. {max(v):8.2f}")
    print(f"\nwrote {out_path}  ({os.path.getsize(out_path)/1048576:.0f} MB)")

main()
