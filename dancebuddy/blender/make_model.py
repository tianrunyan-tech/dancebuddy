"""Turn an unrigged 3D model (GLB/GLTF/OBJ/FBX, e.g. from Tripo) into a DanceBuddy
character: decimate it, stand it on the floor at a standard size, add a
4-bone spine (root -> lower -> upper -> top) weighted by height, render 4 views.

Run `python -m dancebuddy model MODEL.glb assets/<name>`, or directly:
  blender -b -P dancebuddy/blender/make_model.py -- --model MODEL.glb --out assets/<name>
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_utils import reset_scene, setup_render, setup_stage  # noqa: E402

BONE_NAMES = ["root", "lower", "upper", "top"]
BONE_SPLITS = [0.0, 0.18, 0.45, 0.72, 1.0]


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--engine", default="auto")
    p.add_argument("--height", type=float, default=1.9, help="target height in metres")
    p.add_argument("--max-width", type=float, default=1.6)
    p.add_argument("--faces", type=int, default=80000, help="decimate to about this many faces")
    p.add_argument("--yaw", type=float, default=0.0, help="extra rotation (degrees) if the model does not face the camera")
    return p.parse_args(argv)


def import_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise SystemExit(f"unsupported model format {ext}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh found in the model")
    # drop any rig that came with the file: we add our own
    for o in list(bpy.context.scene.objects):
        if o.type == "ARMATURE":
            for m in meshes:
                for mod in list(m.modifiers):
                    if mod.type == "ARMATURE":
                        m.modifiers.remove(mod)
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
        m.vertex_groups.clear()
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    body = bpy.context.view_layer.objects.active
    body.parent = None
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    body.name = "Body"
    return body


def normalise(body, a):
    if a.yaw:
        body.rotation_mode = "XYZ"
        body.rotation_euler.z = math.radians(a.yaw)
        bpy.ops.object.transform_apply(rotation=True)
    n_faces = len(body.data.polygons)
    if n_faces > a.faces:
        mod = body.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = a.faces / n_faces
        bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"faces: {n_faces} -> {len(body.data.polygons)}")
    co = [body.matrix_world @ v.co for v in body.data.vertices]
    lo = Vector((min(c.x for c in co), min(c.y for c in co), min(c.z for c in co)))
    hi = Vector((max(c.x for c in co), max(c.y for c in co), max(c.z for c in co)))
    h, w = hi.z - lo.z, hi.x - lo.x
    s = min(a.height / h, a.max_width / w)
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for v in body.data.vertices:
        v.co = (v.co - centre) * s
    body.data.update()
    return h * s, w * s


def rig(body, height):
    arm_data = bpy.data.armatures.new("Rig")
    rig_obj = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig_obj)
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.mode_set(mode="EDIT")
    prev, bones = None, []
    for name, f0, f1 in zip(BONE_NAMES, BONE_SPLITS[:-1], BONE_SPLITS[1:]):
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = Vector((0, 0, f0 * height)), Vector((0, 0, f1 * height))
        if prev is not None:
            b.parent, b.use_connect = prev, True
        prev = b
        bones.append((name, f0 * height, f1 * height))
    bpy.ops.object.mode_set(mode="OBJECT")
    centres = [(z0 + z1) / 2 for _, z0, z1 in bones]
    groups = {n: body.vertex_groups.new(name=n) for n, _, _ in bones}
    per_bone = {n: ([], []) for n, _, _ in bones}
    for v in body.data.vertices:
        z = v.co.z
        ws = []
        for i, c in enumerate(centres):
            half = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]) / 2 or 0.5
            w = max(0.0, 1 - abs(z - c) / half)
            if (i == 0 and z < c) or (i == len(centres) - 1 and z > c):
                w = 1.0
            ws.append(w)
        s = sum(ws) or 1.0
        for (n, _, _), w in zip(bones, ws):
            if w > 0:
                groups[n].add([v.index], w / s, "REPLACE")
    arms = add_arms(body, rig_obj, bones, height, groups)
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig_obj
    body.parent = rig_obj
    for pb in rig_obj.pose.bones:
        pb.rotation_mode = "XYZ"
    return bones, arms


def add_arms(body, rig_obj, bones, height, groups):
    """Find limbs sticking out sideways (arms, wings, flippers) and give each one a bone.

    Heuristic: in the middle 15-75% of the height, vertices beyond 70% of the
    model's half-width on one side are "arm"; the arm bone runs from the
    innermost of those vertices (the shoulder) to the outermost (the hand)."""
    vs = body.data.vertices
    xmax = max(abs(v.co.x) for v in vs)
    arms = []
    for side, sign in (("L", 1), ("R", -1)):
        pts = [v for v in vs if sign * v.co.x > 0.7 * xmax and 0.15 * height < v.co.z < 0.75 * height]
        if len(pts) < 50:
            continue
        # the arm must stick out past the body above and below it, otherwise it is just a wide body
        band_lo = min(v.co.z for v in pts)
        band_hi = max(v.co.z for v in pts)
        if band_hi - band_lo > 0.5 * height:
            continue
        tip = max(pts, key=lambda v: sign * v.co.x).co.copy()
        sx = 0.7 * xmax * sign
        zc = sum(v.co.z for v in pts) / len(pts)
        shoulder = Vector((sx * 0.75, 0, zc))
        parent = next(n for n, z0, z1 in bones if z0 <= zc <= z1) if zc < bones[-1][2] else bones[-1][0]
        name = f"arm_{side}"
        bpy.context.view_layer.objects.active = rig_obj
        bpy.ops.object.mode_set(mode="EDIT")
        eb = rig_obj.data.edit_bones.new(name)
        eb.head, eb.tail = shoulder, Vector((tip.x, tip.y, tip.z))
        eb.roll = 0
        eb.parent = rig_obj.data.edit_bones[parent]
        bpy.ops.object.mode_set(mode="OBJECT")
        g = body.vertex_groups.new(name=name)
        arm_ids = set()
        for v in pts:
            # ramp the weight in over the first part of the arm so the shoulder bends smoothly
            t = min(1.0, max(0.0, (sign * v.co.x - 0.7 * xmax) / (0.15 * xmax)))
            if t <= 0:
                continue
            g.add([v.index], t, "REPLACE")
            arm_ids.add(v.index)
            for other in groups.values():
                try:
                    w = other.weight(v.index)
                    other.add([v.index], w * (1 - t), "REPLACE")
                except RuntimeError:
                    pass
        arms.append({"name": name, "parent": parent, "side": "left" if side == "L" else "right",
                     "head": [round(c, 3) for c in shoulder], "tail": [round(c, 3) for c in tip],
                     "length": round((tip - shoulder).length, 3), "limb": "arm"})
        print(f"found {name}: {len(arm_ids)} vertices, parent {parent}")
    return arms


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    reset_scene()
    body = import_model(os.path.abspath(a.model))
    height, width = normalise(body, a)
    bones, arms = rig(body, height)
    setup_stage(target_z=height / 2, height=height)
    setup_render(a.engine, 512, 512)
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")

    views_dir = out + "_views"
    os.makedirs(views_dir, exist_ok=True)
    scene = bpy.context.scene
    cam = scene.camera
    target = Vector((0, 0, height / 2))
    dist = (cam.location - target).length
    views = {}
    for view, ang in [("front", 0), ("three_quarter", 40), ("side", 90), ("back", 180)]:
        r = math.radians(ang)
        cam.location = Vector((dist * math.sin(r), -dist * math.cos(r), cam.location.z))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(views_dir, f"{view}.png")
        bpy.ops.render.render(write_still=True)
        views[view] = os.path.relpath(scene.render.filepath, os.path.dirname(out))
    rig_info = {
        "name": os.path.basename(out),
        "kind": "3D model with an automatic 4-bone spine rig",
        "source_model": os.path.basename(a.model),
        "height_m": round(height, 3), "width_m": round(width, 3),
        "root_bone": "root",
        "bones": [{"name": n, "parent": bones[i - 1][0] if i else None, "head_z": round(z0, 3), "tail_z": round(z1, 3),
                   "length": round(z1 - z0, 3)} for i, (n, z0, z1) in enumerate(bones)] + arms,
        "channels_per_bone": ["lean", "twist", "sway"],
        "root_only_channels": ["lift", "squash"],
        "limits": {"lean": 35, "twist": 45, "sway": 35, "lift": 0.35, "squash": 0.35,
                   **{f"{a['name']}.{c}": 70 for a in arms for c in ("lean", "sway", "twist")}},
        "notes": ("the spine bends the whole body; hair and other parts follow the spine bone at their height. "
                  + ("Arm bones (arm_L, arm_R) point sideways: lean raises (+) or lowers (-) the arm, sway swings it "
                     "forward/back, twist rolls it around its own length." if arms else "No limbs were found.")),
        "views": views,
    }
    with open(out + "_rig.json", "w") as f:
        json.dump(rig_info, f, indent=1)
    print(f"wrote {out}.blend and {out}_rig.json")


main()
