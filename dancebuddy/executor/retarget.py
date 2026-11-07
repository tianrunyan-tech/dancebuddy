"""Copy a dancer onto a character, following the character's motion spec.

What the body can do is copied frame by frame; what it cannot is translated (assets/<name>_motion_spec.json):
  * arms   : one bendable bone per arm: it points from the shoulder toward the hand and curves from the dancer's
             upper-arm direction to her forearm direction, keeping its width; kept in front of the hair and body
  * body   : the dancer's torso turn / side tilt / forward lean drive the whole body around its base, at full size
             and smoothly compressed near the character's limits (never cut flat, so peaks keep their timing)
  * legs   : no legs -> every jump, kick, step, squat and move across is translated (legs.py) into hops,
             squash and stretch, leans and slides of the body
  * hair   : not animated: it moves with the head
  * groove : a beat-synced wiggle of the soft body (sway, twist, upper-body counter-bend, a squash pulse)

Output is a curves.json with quaternion channels (`<bone>.quat`, [w, x, y, z] in the bone's rest frame) plus
`root.loc` (slide across, hop) and `root.squash`, rendered by dancebuddy/blender/render_curves.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancebuddy.perception.beats import smooth
from dancebuddy.perception.video import L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR

# the character's range (motion spec "can_do"): the dancer is copied at full size and compressed softly near these
BODY = {"turn_max": 180.0,       # it can turn all the way round (motion spec)
        "side_max": 25.0, "lean_max": 20.0,
        "turn_quiet_deg": 30.0}  # turns below this (by the shoulder width) are softened: there it is mostly noise
# the body is a soft bag that keeps its volume (motion spec body_soft): part of every sideways tilt is a bend of the
# body itself (a C curve from the base), a hip lift bends it toward the lifted hip, hips swaying under a still torso
# make an S, and the pressed bag bulges: the lower body gets wider and shorter as it bends
SOFT = {"bend_share": 0.6,          # part of the side tilt shown as a bend (the rest tilts it on its base)
        "hip_bend": 0.8,            # bend (deg) per deg of the dancer's hip tilt
        "sway_bend": 25.0,          # S-curve bend (deg) per torso length of the hips' sideways shift
        "bend_max": 22.0,           # per body bone
        "bulge_per_deg": 0.005}     # widening of the lower body per deg of bend (height shrinks to keep the volume)


def soft_limit(x: np.ndarray, limit: float) -> np.ndarray:
    """Full size for small values, smoothly compressed toward the limit (peaks keep their timing)."""
    return limit * np.tanh(np.asarray(x, float) / limit)
# arm directions: in-plane from the image (reliable), depth from the 3D estimate (monocular depth is not)
ARM_DEPTH_GAIN = 0.4
# groove: a beat-synced wiggle of the soft body (sway and twist a quarter beat apart, upper body bending the
# other way a moment later), backing off while the dancer's own body moves a lot
GROOVE = {"tilt_deg": 5.0, "twist_deg": 7.0, "bend_deg": 5.0, "bend_delay_s": 0.08, "pulse_squash": 0.03,
          "busy_deg": 15.0}
# style from the spec's "tends_to": arms lead and the body follows, soft arms, squash low / stretch high
STYLE = {"straight_above_deg": 120,         # raised arms straighten out (reference: arms up are straight)
         "bend_share": 0.8,                 # how much of the dancer's elbow bend the soft arm takes, as one arc
         "bend_max_deg": 160.0,             # the most the arm curls (as one round arc)
         "clear_from": 0.12,                # the arm from here to the hand stays out of the body...
         "clear_step_deg": 12.0,            # ...turned away from it at most this much at a time...
         "clear_steps": 12,                 # ...at most this many times...
         "rest_match_to_deg": 110.0,        # the hanging-pose difference fades out by this arm raise (deg)
         "clear_smooth_s": 0.08,            # the corrected arm is smoothed over this long
         "clear_tolerance": 0.005,
         "touch_share": 0.75,               # where the agent sets an arm's pose (moves.py), the arm may lie on the
                                            # body (hands at the chest, an arm across it): this much less clearance          # ...until it sinks in less than this (body heights): plush gives
         "elbow_from_deg": 40.0,            # a bend this big counts as bent (for the fists)...
         "elbow_full_deg": 90.0,            # ...fully from here
         # the shoulder is a socket that slides up the side of the body as the arm rises (reference: a raised
         # arm starts beside the face, a hanging arm beside the chest), so raised hands get above the head
         "socket_slide": 0.2,               # how far the seam slides up the body for a vertical arm (body heights):
                                            # enough for raised hands to reach above the head (motion spec)
         "root_share": 0.5,                 # part of the arm's turn taken at the seam, so the root turns with the
                                            # arm instead of folding (the shoulder pivots just inside the seam)
         "socket_slide_from_deg": 80,
         # hands ball up into fists when the arm is raised and bent, and open when it reaches out or drops
         "fist_from_deg": 95.0,             # the forearm points this far up (0 down, 90 level) before the hand closes
         "fist_full_deg": 140.0,            # ...and is a full fist from here, if the arm is also bent
         "fist_straight": 0.6,              # how far a raised but straight arm closes its hand
         "forearm_lag_s": 0.07,             # the end of the arm trails the upper arm a little
         "follow_tilt_deg": 6.0,            # the body leans toward the higher arm...
         "follow_delay_s": 0.4,             # ...about this long after the arms move
         }


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def to_blender(p: np.ndarray) -> np.ndarray:
    """MediaPipe world (x screen-right, y down, z negative toward the camera) -> Blender (x right, y away, z up)."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def image_aspect(kp: dict) -> float:
    """Width / height of the video, estimated from the body (keypoints.npz has no frame size): the shoulder
    width / torso length ratio must be the same in the 3D estimate and in the image."""
    w, im = kp["world"], kp["image"]
    sw_w = np.linalg.norm(w[:, L_SH, :2] - w[:, R_SH, :2], axis=1)
    th_w = np.abs(((w[:, L_HIP] + w[:, R_HIP]) / 2 - (w[:, L_SH] + w[:, R_SH]) / 2)[:, 1])
    sw_i = np.abs(im[:, L_SH, 0] - im[:, R_SH, 0])
    th_i = np.abs(((im[:, L_HIP] + im[:, R_HIP]) / 2 - (im[:, L_SH] + im[:, R_SH]) / 2)[:, 1])
    facing = sw_i >= np.percentile(sw_i, 60)  # frames where the dancer faces the camera
    r = (sw_w / np.maximum(th_w, 1e-6) * th_i / np.maximum(sw_i, 1e-6))[facing]
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else 1.0


def limb_dirs(kp: dict, w: np.ndarray, a: int, b: int, aspect: float) -> np.ndarray:
    """(N,3) direction a->b in Blender axes: x and z from the image, y (depth) from the 3D estimate, damped."""
    im = kp["image"]
    dx = (im[:, b, 0] - im[:, a, 0]) * aspect
    dz = -(im[:, b, 1] - im[:, a, 1])
    v3 = w[:, b] - w[:, a]
    scale = np.linalg.norm(np.c_[dx, dz], axis=1) / np.maximum(np.linalg.norm(v3[:, [0, 2]], axis=1), 1e-6)
    dy = v3[:, 1] * scale * ARM_DEPTH_GAIN
    d = np.c_[dx, dy, dz]
    sig = 0.05 * float(kp.get("fps", 30.0))  # pose noise: smooth over ~50 ms whatever the frame rate
    d = np.stack([smooth(d[:, k], sig) for k in range(3)], -1)
    return _norm(d)


def body_turn(kp: dict, T: np.ndarray) -> np.ndarray:
    """(N,) degrees the dancer has turned, + = her left side away from the camera; up to 180 (her back).

    How far: from how wide her shoulders look in the picture compared with facing the camera (monocular depth is
    not reliable enough to measure it); which way: from the 3D estimate (only its sign is needed). Past side-on
    (shoulders swapped in the picture) counts only when both shoulders are clearly seen for a few frames in a
    row; otherwise the turn stops at side-on. The hips' width is a second reading (raised arms narrow the
    shoulders). Near facing the camera the width barely changes with the turn, so
    small wobbles there (pose noise, arm swings) are not read as turns."""
    im, vis = kp["image"], kp.get("visibility")
    fps = float(kp.get("fps", 30.0))
    aspect = image_aspect(kp)
    sw = (im[:, L_SH, 0] - im[:, R_SH, 0]) * aspect
    torso = np.abs((im[:, L_HIP, 1] + im[:, R_HIP, 1] - im[:, L_SH, 1] - im[:, R_SH, 1]) / 2)
    hw = (im[:, L_HIP, 0] - im[:, R_HIP, 0]) * aspect
    r = smooth(sw / np.maximum(torso, 1e-6), 0.06 * fps)
    rh = smooth(hw / np.maximum(torso, 1e-6), 0.06 * fps)
    r = r / max(float(np.percentile(r, 90)), 1e-6)
    rh = rh / max(float(np.percentile(rh, 90)), 1e-6)
    # a raised arm lifts and pulls in its shoulder, so the shoulders look narrower than the turn: facing the
    # camera, take whichever of shoulders and hips reads the smaller turn
    r = np.where((r > 0) & (rh > 0), np.maximum(r, rh), r)
    front = 1.0
    seen = np.ones(len(r), bool) if vis is None else (np.minimum(vis[:, L_SH], vis[:, R_SH]) >= 0.8)
    back = (r < 0) & seen
    run = np.zeros(len(r), bool)  # swapped and seen for at least 5 frames in a row
    i = 0
    while i < len(r):
        if back[i]:
            j = i
            while j < len(r) and back[j]:
                j += 1
            run[i:j] = (j - i) >= 5
            i = j
        else:
            i += 1
    ratio = np.where(run, r / front, np.maximum(r / front, 0.0))
    mag = np.degrees(np.arccos(np.clip(ratio, -1, 1)))
    g = np.clip(mag / BODY["turn_quiet_deg"], 0, 1)
    mag = mag * g * g * (3 - 2 * g)                           # soft dead zone near facing the camera
    sign = np.sign(smooth(T[:, 1, 0], 0.1 * fps) + 1e-9)      # which shoulder is farther from the camera
    # the depth sign flips on back views, so only read it near facing the camera and hold it through the turn
    n = len(mag)
    held = sign.copy()
    for i in range(1, n):
        if mag[i] > 60.0 and mag[i - 1] > 20.0:
            held[i] = held[i - 1]
    turn = mag * held
    # full spin: she comes back to the front from the other side, so keep turning past 180
    offset, i = 0.0, 0
    while i < n:
        if mag[i] > 60.0:
            j = i
            while j < n and mag[j] > 60.0:
                j += 1
            s_in = held[i]
            peak = i + int(np.argmax(mag[i:j]))
            if j < n and sign[j] != s_in and mag[peak] >= 100.0:
                turn[i:peak] += offset
                turn[peak:j] = s_in * (360.0 - mag[peak:j]) + offset
                offset += s_in * 360.0
            else:
                turn[i:j] += offset
            i = j
        else:
            turn[i] += offset
            i += 1
    return smooth(turn, 0.05 * fps)


def hip_tilt(kp: dict, aspect: float) -> np.ndarray:
    """(N,) degrees the dancer's pelvis is tilted in the picture, + = her left hip higher (a hip lift)."""
    im = kp["image"]
    d = im[:, L_HIP, :2] - im[:, R_HIP, :2]
    ang = np.degrees(np.arctan2(-d[:, 1], np.abs(d[:, 0]) * aspect + 1e-6))
    return smooth(ang - np.median(ang), 2.0)


def torso_frames(w: np.ndarray) -> np.ndarray:
    """(N,3,3) torso orientation; columns = dancer's left, back, up (the character's rest axes X, Y, Z)."""
    sh, hip = (w[:, L_SH] + w[:, R_SH]) / 2, (w[:, L_HIP] + w[:, R_HIP]) / 2
    z = _norm(sh - hip)
    x = w[:, L_SH] - w[:, R_SH] + 0.5 * (w[:, L_HIP] - w[:, R_HIP])
    x = _norm(x - (x * z).sum(1, keepdims=True) * z)
    y = np.cross(z, x)
    return np.stack([x, y, z], -1)


def _align(target: np.ndarray, current: np.ndarray) -> Rot:
    """Smallest rotation taking direction `current` onto `target`."""
    a, b = _norm(current), _norm(target)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-8:
        return Rot.identity() if c > 0 else Rot.from_rotvec(np.pi * _norm(np.cross(a, [0.0, 0.0, 1.0])))
    return Rot.from_rotvec(axis / s * np.arctan2(s, c))


def _local(M: np.ndarray, parent_c: Rot, c: Rot) -> list[float]:
    """Pose rotation of a bone with rest matrix M whose parent chain is rotated by parent_c (armature space),
    so that the bone ends up rotated by c (armature space): L = M^T parent_c^T c M."""
    m = Rot.from_matrix(M)
    q = (m.inv() * parent_c.inv() * c * m).as_quat()  # x, y, z, w
    return [round(float(q[3]), 5), round(float(q[0]), 5), round(float(q[1]), 5), round(float(q[2]), 5)]


def keep_clear(u: np.ndarray, sign: int) -> np.ndarray:
    """Keep an arm direction (character rest axes: x left, y back, z up) in front of the hair and the body."""
    u = u.copy()
    u[1] = min(u[1], 0.25)                       # never far behind the body
    if u[2] > 0:
        u[1] = min(u[1], -0.35 * u[2])           # raised arms pass just in front of the hair (not foreshortened)
        u[0] += sign * 0.15 * u[2]               # and open slightly outward so they read beside the head
    if sign * u[0] < 0.2:
        u[1] = min(u[1], -0.5)                   # arms crossing the body go in front of it
    return _norm(u)


def match_rest(d: np.ndarray, rest_dir: np.ndarray) -> np.ndarray:
    """The dancer's arm hangs straight down beside a slim body; the character's arm hangs out and forward around
    its round body (its rest pose). Turn a direction by that difference, fully for a hanging arm and fading out
    as the arm rises, so the dancer's swings start from the character's own hanging pose instead of from inside
    its body."""
    off = _align(_norm(np.asarray(rest_dir, float)), np.array([0.0, 0.0, -1.0])).as_rotvec()
    k = float(np.clip(1 - raise_deg(_norm(d)) / STYLE["rest_match_to_deg"], 0, 1))
    return Rot.from_rotvec(off * k * k * (3 - 2 * k)).apply(d)


def raise_deg(u: np.ndarray) -> float:
    """0 = hanging down, 90 = horizontal, 180 = straight up (direction in character rest axes)."""
    return float(np.degrees(np.arccos(np.clip(-u[2], -1, 1))))


def socket_turn(u: np.ndarray, sign: int) -> Rot:
    """Turn of the hidden shoulder bone (armature space) that slides the arm's root up the side of the body as
    the arm rises: it swings the root upward around a pivot inside the body."""
    a = raise_deg(_norm(u))
    k = float(np.clip((a - STYLE["socket_slide_from_deg"]) / (175 - STYLE["socket_slide_from_deg"]), 0, 1))
    k = k * k * (3 - 2 * k)
    return Rot.from_rotvec([0.0, -sign * np.radians(STYLE["socket_slide_deg"]) * k, 0.0])


def socket_rot(M: dict, names: list[str], u: np.ndarray, f: np.ndarray, sign: int) -> Rot:
    """The hidden shoulder bone turns with the arm (a share of its turn), so a big arm turn is not all taken at
    the arm's root, where it would fold the arm."""
    u_c = keep_clear(_norm(u), sign)
    target = keep_clear(_norm(u_c + _norm(f)), sign)
    rest = _norm(np.array(M[f"{names[-1]}__tail"]) - np.array(M[f"{names[0]}__head"]))
    return Rot.from_rotvec(_align(target, rest).as_rotvec() * STYLE["root_share"])


def socket_offset(u: np.ndarray, sign: int, height: float) -> np.ndarray:
    """Where the seam slides as the arm rises (armature axes, metres): up the body side, a little inward as the
    body narrows, and a little forward so the arm stays in front of the hair."""
    a = raise_deg(keep_clear(_norm(u), sign))
    k = float(np.clip((a - STYLE["socket_slide_from_deg"]) / (175 - STYLE["socket_slide_from_deg"]), 0, 1))
    dz = STYLE["socket_slide"] * height * k * k * (3 - 2 * k)
    return np.array([-sign * 0.3 * dz, -0.15 * dz, dz])


def _hand_roll(n: np.ndarray, d: np.ndarray) -> float:
    """Turn (radians, around the hand's direction d) that brings the hand's flat face n toward the camera (-Y):
    palm or back, whichever is closer. 0 when the hand points at the camera."""
    n_p, cam = n - d * (n @ d), np.array([0.0, -1.0, 0.0]) - d * (-d[1])
    if np.linalg.norm(n_p) < 1e-6 or np.linalg.norm(cam) < 0.2:
        return 0.0
    n_p, cam = _norm(n_p), _norm(cam)
    if n_p @ cam < 0:
        cam = -cam
    return float(np.arctan2(np.cross(n_p, cam) @ d, n_p @ cam))


BBONE_HANDLE = 0.390464  # Blender: a B-bone handle reaches this x the bone length x its ease (rest ease 1)


def arm_pose(M: dict, name: str, u: np.ndarray, f: np.ndarray, sign: int,
             parent: Rot | None = None, scale: float = 1.0, face: list | None = None,
             info: dict | None = None) -> tuple[list, list, float]:
    """Pose of a one-bone soft arm (a B-bone): its rotation [w, x, y, z], its curve [curve in x, in z, out x,
    out z, roll in, roll out, ease in, ease out] and how much its chord shortens (bone y-scale).

    The arm is one bendable tube that keeps its width and length: the bone points from the shoulder toward the
    hand, the curve is a circular arc whose start leans toward the dancer's upper arm and whose end comes from
    the forearm's direction (the elbow bend spread evenly along the arm, so it stays round), and the hand's flat
    face turns toward the camera through a twist that grows along the arm (none at the seam)."""
    parent = parent if parent is not None else Rot.identity()
    u = keep_clear(_norm(u), sign)
    f = _norm(f)
    if f[1] > 0.3:
        f = _norm(f - np.array([0, f[1] - 0.3, 0]))
    k = float(np.clip((raise_deg(u) - STYLE["straight_above_deg"]) / 40.0, 0, 1))
    f = _norm((1 - k) * f + k * u)                   # raised arms are straight
    target = keep_clear(_norm(u + f), sign)          # shoulder -> hand
    head, tail = np.array(M[f"{name}__head"], float), np.array(M[f"{name}__tail"], float)
    length = float(np.linalg.norm(tail - head))
    c = _align(target, M[name][:, 1])                # the arm with its rest shape, turned as a whole
    if scale != 1.0:                                 # the agent's choice: exaggerate or soften the arm move
        c = Rot.from_rotvec(c.as_rotvec() * scale)
    frame = c * Rot.from_matrix(M[name])             # bone axes -> armature
    chord = frame.apply([0.0, 1.0, 0.0])
    # the arm is a thick plush tube: it bends as one round arc (the same curvature all along, so the inside of
    # the bend never folds or creases), and keeps its length: the straight distance from shoulder to hand
    # shortens as the arc closes (bone y-scale), the arc itself stays as long as the arm
    lu, lf = frame.inv().apply(u), frame.inv().apply(f)
    a_in = float(np.arctan2(np.hypot(lu[0], lu[2]), lu[1]))
    a_out = float(np.arctan2(np.hypot(lf[0], lf[2]), lf[1]))
    theta = min((a_in + a_out) * STYLE["bend_share"] * scale, np.radians(STYLE["bend_max_deg"]))
    a_out = theta / 2
    chord_scale, cin, cout, ease = 1.0, np.zeros(2), np.zeros(2), 0.0
    if theta > 1e-3:
        chord_scale = float(np.sin(theta / 2) / (theta / 2))
        radius = length / (2 * np.sin(theta / 2))                     # of the arc over the rest-length chord
        handle = 4 / 3 * np.tan(theta / 4) * radius                   # Bezier handles of a circular arc
        ease = float(handle * np.cos(theta / 2) / (BBONE_HANDLE * length) - 1)
        off = handle * np.sin(theta / 2)
        # the arc bulges away from the shoulder->hand line on the side opposite the elbow's inside: its start
        # leans toward the upper arm, its end comes in from the forearm's side
        side = np.array([lu[0] - lf[0], lu[2] - lf[2]])
        if np.linalg.norm(side) < 1e-6:
            side = np.array([lu[0], lu[2]])
        if np.linalg.norm(side) > 1e-6:
            d = side / np.linalg.norm(side)
            cin, cout = off * d, off * d
    if info is not None:                              # the arc in armature space, for keeping it out of the body
        bulge = frame.apply([cin[0], 0.0, cin[1]]) if theta > 1e-3 else np.zeros(3)
        info.update(chord=chord, length=length, theta=theta,
                    bulge=_norm(bulge) if np.linalg.norm(bulge) > 1e-9 else np.zeros(3))
    roll = 0.0
    if face is not None:                              # the hand's direction and face at the end of the curve
        ax = np.cross(chord, f)
        bend_end = Rot.from_rotvec(_norm(ax) * a_out) if np.linalg.norm(ax) > 1e-6 else Rot.identity()
        roll = _hand_roll(bend_end.apply(c.apply(np.asarray(face, float))), bend_end.apply(chord))
    q = _local(M[name], parent, c)
    curve = [round(float(x), 5) for x in (cin[0], cin[1], cout[0], cout[1], 0.0, roll, ease, ease)]
    return q, curve, round(chord_scale, 5)


def arm_centre_line(head: np.ndarray, info: dict, ts: np.ndarray) -> np.ndarray:
    """Points along a posed arm (armature space) at fractions ts of its length: a circular arc of the arm's
    length from its head, bulging toward info['bulge']."""
    L, th, ch, b = info["length"], info["theta"], info["chord"], info["bulge"]
    if th < 1e-3:
        return head + np.outer(ts * L, ch)
    R = L / th
    phi = (ts - 0.5) * th
    half_chord = R * np.sin(th / 2)
    return head + np.outer(half_chord + R * np.sin(phi), ch) + np.outer(R * (np.cos(phi) - np.cos(th / 2)), b)


def body_depth(pts: np.ndarray, shape: dict, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """How deep (m, + = inside) points sink into the body (its rest shape, rig.json body_shape), with a margin
    (the arm's thickness), and the outward direction at each point."""
    rad = np.asarray(shape["radius"], float)
    nz, na = rad.shape
    cx, cy = shape["centre"]
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    zf = np.clip(pts[:, 2] / shape["top"] * nz - 0.5, 0, nz - 1)
    af = ((ang + np.pi) / (2 * np.pi) * na - 0.5) % na
    z0, a0 = np.floor(zf).astype(int), np.floor(af).astype(int)
    z1, a1 = np.minimum(z0 + 1, nz - 1), (a0 + 1) % na
    wz, wa = zf - z0, af - a0
    body = ((1 - wz) * ((1 - wa) * rad[z0, a0] + wa * rad[z0, a1]) + wz * ((1 - wa) * rad[z1, a0] + wa * rad[z1, a1]))
    above = (pts[:, 2] > shape["top"]) | (pts[:, 2] < 0)
    depth = np.where(above, -1.0, body + margin - r)
    out = np.stack([dx, dy, np.zeros_like(dx)], -1) / np.maximum(r, 1e-6)[:, None]
    return depth, out


def elbow_amount(u: np.ndarray, f: np.ndarray) -> float:
    """0 for a small bend .. 1 for a big bend of the dancer's elbow."""
    bend = float(np.degrees(np.arccos(np.clip(_norm(u) @ _norm(f), -1, 1))))
    return float(np.clip((bend - STYLE["elbow_from_deg"]) / (STYLE["elbow_full_deg"] - STYLE["elbow_from_deg"]), 0, 1))


def fist_amount(u: np.ndarray, f: np.ndarray) -> float:
    """How far the hand is balled up (0 open .. 1 fist), from the motion spec: fists for raised, bent arms, open
    hands for reaching out or hanging. Raised = the forearm points up; a straight raised arm closes partly."""
    up = (raise_deg(_norm(f)) - STYLE["fist_from_deg"]) / (STYLE["fist_full_deg"] - STYLE["fist_from_deg"])
    up = float(np.clip(up, 0, 1))
    up = up * up * (3 - 2 * up)
    return up * (STYLE["fist_straight"] + (1 - STYLE["fist_straight"]) * elbow_amount(u, f))


def beat_phase(n: int, beats: list[int]) -> np.ndarray:
    """Continuous beat count per frame (k at the k-th beat), extrapolated with the median beat period."""
    b = sorted(int(x) for x in beats)
    if len(b) < 2:
        return np.arange(n) / 15.0
    period = float(np.median(np.diff(b)))
    xs = np.r_[b[0] - period * 50, b, b[-1] + period * 50]
    ks = np.r_[-50, np.arange(len(b)), len(b) - 1 + 50]
    return np.interp(np.arange(n), xs, ks)


def groove(n: int, fps: float, beats: list[int] | None, busy: np.ndarray) -> dict:
    """Beat-synced wiggle: side sway (one side per beat), twist a quarter beat later, an upper-body counter-bend,
    and a small squash pulse on each beat; scaled down while the body is busy with the dancer's own moves."""
    if not beats:
        z = np.zeros(n)
        return {"tilt": z, "twist": z, "bend": z, "pulse": z}
    ph = beat_phase(n, beats)
    amp = 1.0 - 0.7 * np.clip(busy / GROOVE["busy_deg"], 0, 1)
    sway = np.sin(np.pi * ph)
    tilt = GROOVE["tilt_deg"] * amp * sway
    twist = GROOVE["twist_deg"] * amp * np.sin(np.pi * ph - np.pi / 2)
    d = max(1, int(round(GROOVE["bend_delay_s"] * fps)))
    bend = -GROOVE["bend_deg"] * np.r_[np.zeros(d), (amp * sway)[:-d]]
    frac = ph % 1.0
    pulse = GROOVE["pulse_squash"] * amp * np.exp(-0.5 * (np.minimum(frac, 1 - frac) / 0.12) ** 2)
    return {"tilt": tilt, "twist": twist, "bend": bend, "pulse": pulse}


def retarget(kp: dict, rig: dict, signals: dict | None = None, beats: list[int] | None = None,
             plan: dict | None = None, score: dict | None = None) -> dict:
    """plan: the agent's decisions (actions/plan.py); None = the plain copy. score: the dance score (move units)."""
    from dancebuddy.actions.plan import emphasis_curves, knob_curves, leg_overrides
    fps = float(kp["fps"])
    w = to_blender(kp["world"])
    w = np.stack([np.stack([smooth(w[:, j, k], 1.5) for k in range(3)], -1) for j in range(w.shape[1])], 1)
    n = len(w)
    T = torso_frames(w)
    bones = {b["name"]: b for b in rig["bones"]}
    M = {k: np.array(b["matrix"], float) for k, b in bones.items()}
    M.update({f"{k}__head": b["head"] for k, b in bones.items()})
    M.update({f"{k}__tail": b["tail"] for k, b in bones.items()})
    rest_dir = {k: M[k][:, 1] for k in M if "__" not in k}  # bone axis = local Y

    if signals is None:
        from dancebuddy.perception.signals import compute_signals
        signals = compute_signals(kp)
    from dancebuddy.executor.legs import translate as translate_legs
    legs = translate_legs(kp, signals, leg_overrides(plan))
    height = float(rig.get("height_m", 1.4))
    knob = knob_curves(plan, score, n, fps)
    from dancebuddy.actions.moves import curves as move_curves, parse as parse_moves
    mv = move_curves(parse_moves((plan or {}).get("moves", "")), n, fps)  # the agent's own moves
    emph = emphasis_curves({"emphasis": [*(plan or {}).get("emphasis", []), *mv["emphasis"]]}, n, fps, height)

    # ---- body: turn, side tilt, forward lean of the dancer's torso, scaled to the character
    X, Z = T[:, :, 0], T[:, :, 2]
    turn = body_turn(kp, T)                                  # + = dancer's left side turns away from the camera
    side, lean = [], []
    for i in range(n):
        z = Rot.from_euler("z", -turn[i], degrees=True).apply(Z[i])
        side.append(np.degrees(np.arctan2(z[0], z[2])))       # + = tilt toward the character's left (+X)
        lean.append(np.degrees(np.arctan2(-z[1], z[2])))      # + = forward (toward the camera, -Y)
    side = smooth(np.array(side), 1.0) * knob["body"] + legs["tilt"] + emph["tilt"] + mv["body"]["tilt"]
    lean = soft_limit(smooth(np.array(lean), 1.0) * knob["body"] + mv["body"]["lean"], BODY["lean_max"])
    if BODY["turn_max"] >= 180.0:  # can spin all the way round: scale only small turns
        turn = turn + (knob["body"] - 1.0) * soft_limit(turn, 60.0) + emph["twist"] + mv["body"]["turn"]
    else:
        turn = soft_limit(turn * knob["body"] + emph["twist"] + mv["body"]["turn"], BODY["turn_max"])

    # arm directions (image-based in the picture plane), in Blender axes
    aspect = image_aspect(kp)
    upper = {"L": limb_dirs(kp, w, L_SH, L_EL, aspect), "R": limb_dirs(kp, w, R_SH, R_EL, aspect)}
    fore = {"L": limb_dirs(kp, w, L_EL, L_WR, aspect), "R": limb_dirs(kp, w, R_EL, R_WR, aspect)}

    # arms lead, the body follows: lean toward the higher arm a moment later
    arm_up = {k: np.array([(T[i].T @ upper[k][i])[2] for i in range(n)]) for k in ("L", "R")}
    delay = int(round(STYLE["follow_delay_s"] * fps))
    diff = arm_up["L"] - arm_up["R"]
    follow = np.r_[np.full(delay, diff[0]), diff[:n - delay]] if delay < n else np.zeros(n)
    side = soft_limit(side + STYLE["follow_tilt_deg"] * smooth(follow, 3.0), BODY["side_max"])
    busy = np.maximum.reduce([np.abs(side), np.abs(lean), np.abs(turn) / 2])
    gv = groove(n, fps, beats, busy)
    gv = {k: v * knob["groove"] for k, v in gv.items()}
    side = side + gv["tilt"]
    turn = turn + gv["twist"]
    # ---- soft body: split the side tilt into a tilt on the base and a bend of the body (C), add the hip lift and the
    # hips' sway (S), and let the bent lower body bulge
    bend_c = side * SOFT["bend_share"]
    side = side - bend_c                                       # the rigid part, on the base
    hip_t = hip_tilt(kp, aspect)                               # + = the dancer's left hip higher
    sway = smooth(np.asarray(signals.get("hip_sway", np.zeros(n)), float), 1.5)
    lower_bend = soft_limit(0.5 * bend_c - SOFT["hip_bend"] * hip_t + SOFT["sway_bend"] * sway, SOFT["bend_max"])
    upper_bend_deg = soft_limit(0.5 * bend_c + 1.5 * SOFT["hip_bend"] * hip_t - SOFT["sway_bend"] * sway,
                                SOFT["bend_max"])
    bulge = 1.0 + SOFT["bulge_per_deg"] * (np.abs(lower_bend) + 0.5 * np.abs(upper_bend_deg))
    body = [Rot.from_euler("z", turn[i], degrees=True) * Rot.from_euler("y", side[i], degrees=True)
            * Rot.from_euler("x", lean[i], degrees=True) for i in range(n)]
    # upper body: S-shaped wiggle, plus the dancer's head nods and tilts when the plan asks for them
    nod = soft_limit(smooth(np.asarray(signals.get("head_nod", np.zeros(n)), float), 1.0) * knob["head"] + emph["nod"], 20.0)
    htilt = soft_limit(smooth(np.asarray(signals.get("head_tilt", np.zeros(n)), float), 1.0) * knob["head"], 20.0)
    upper_bend = [Rot.from_euler("y", gv["bend"][i] + htilt[i] + upper_bend_deg[i], degrees=True)
                  * Rot.from_euler("x", nod[i], degrees=True) for i in range(n)]

    ch: dict[str, list] = {k: [] for k in ["root.quat", "root.loc", "root.squash"]}
    ident = Rot.identity()
    for i in range(n):
        ch["root.quat"].append(_local(M["root"], ident, body[i]))
        # root rest frame: x = across (character's left), y = up
        # tilting on the base would push one side of the bottom into the floor: rise like a roly-poly toy
        tilt_up = body[i].apply([0.0, 0.0, 1.0])
        roll = float(rig.get("base_radius_m", 0.0)) * np.sqrt(max(0.0, 1 - tilt_up[2] ** 2))
        ch["root.loc"].append([round(float(legs["shift"][i] * knob["travel"][i] * height), 4),
                               round(float((legs["lift"][i] + emph["lift"][i] + mv["body"]["lift"][i]) * height + roll), 4),
                               0.0])
    ch["root.squash"] = [round(float(x), 4) for x in np.clip(legs["squash"] + gv["pulse"] + emph["squash"]
                                                             + mv["body"]["squash"], -0.1, 0.25)]
    if "body_upper" in M:
        ch["body_upper.quat"] = [_local(M["body_upper"], ident, upper_bend[i]) for i in range(n)]
    if "body_lower" in M:
        ch["body_lower.quat"] = [_local(M["body_lower"], ident, Rot.from_euler("y", lower_bend[i], degrees=True))
                                 for i in range(n)]
        # bone axes: y = up, x = across: wider and as much shorter, so the bag keeps its volume
        ch["body_lower.scale"] = [[round(float(b), 4), round(float(1 / b), 4), 1.0] for b in bulge]

    # ---- arms: copied in the torso frame (the body bones already carry the body rotation)
    clear_report = {}
    for side_name, sign, (sh, el, wr) in (("L", 1, (L_SH, L_EL, L_WR)), ("R", -1, (R_SH, R_EL, R_WR))):
        name = f"arm_{side_name}_1"
        if name not in M:
            continue
        ch[f"{name}.quat"], ch[f"{name}.bbone"], ch[f"{name}.scale"] = [], [], []
        fist = np.zeros(n)
        socket = f"arm_{side_name}_0" if f"arm_{side_name}_0" in M else None
        if socket:
            ch[f"{socket}.quat"] = []
            ch[f"{socket}.loc"] = []
        lag_f = int(round(STYLE["forearm_lag_s"] * fps))
        shape, radius = rig.get("body_shape"), float(rig.get("arm_radius", {}).get(side_name, 0.1))
        ts = np.linspace(STYLE["clear_from"], 1.0, 16)
        # near the seam the arm is sewn onto the body: only its centre must stay out there, the full thickness
        # from a third of the way out
        margin = radius * np.clip((ts - STYLE["clear_from"]) / (0.35 - STYLE["clear_from"]), 0, 1)
        if shape is not None:
            # how close the arm sits to the body in the motion spec's own poses (hanging as modelled, out to the
            # side, raised beside the face) is touching, not sinking in: the body shape is rough near the arm
            rest_dir = _norm(np.asarray(M[f"{name}__tail"], float) - np.asarray(M[f"{name}__head"], float))
            excess = np.zeros(len(ts))
            for d in (rest_dir, np.array([sign, 0.0, 0.0]), _norm(np.array([0.35 * sign, -0.2, 0.95]))):
                c0 = socket_rot(M, [name], d, d, sign) if socket else Rot.identity()
                off = socket_offset(d, sign, height) if socket else np.zeros(3)
                info0 = {}
                arm_pose(M, name, d, d, sign, parent=c0, info=info0)
                p0 = np.asarray(M[f"{socket}__head"], float) if socket else np.zeros(3)
                head0 = p0 + off + c0.apply(np.asarray(M[f"{name}__head"], float) - p0)
                d0, _ = body_depth(arm_centre_line(head0, info0, ts), shape, margin)
                excess = np.maximum(excess, d0)
            margin = margin - excess

        def place_arm(u, f, scale, touch=0.0, name=name, socket=socket, sign=sign, shape=shape, margin=margin, ts=ts):
            """Pose the arm; if any part of it sinks into the body, turn it (upper arm and forearm together,
            around the shoulder) a little at a time away from the body until it is clear (motion spec: arms pass
            in front of the body; a hand on the chest lies on it)."""
            for _ in range(STYLE["clear_steps"] + 1):
                c0 = socket_rot(M, [name], u, f, sign) if socket else Rot.identity()
                off = socket_offset(u, sign, height) if socket else np.zeros(3)
                info = {}
                q, curve, chord = arm_pose(M, name, u, f, sign, parent=c0, scale=scale,
                                           face=rig.get("hand_face", {}).get(side_name), info=info)
                if shape is None:
                    break
                p0 = np.asarray(M[f"{socket}__head"], float) if socket else np.zeros(3)
                head = p0 + off + c0.apply(np.asarray(M[f"{name}__head"], float) - p0)
                pts = arm_centre_line(head, info, ts)
                depth, out = body_depth(pts, shape, margin * (1 - STYLE["touch_share"] * touch))
                k = int(np.argmax(depth))
                if depth[k] <= STYLE["clear_tolerance"] * height:
                    break
                # straight out of the body: against the gradient of the depth (the surface normal)
                e = 0.01
                grad = np.array([(body_depth(pts[k:k + 1] + e * ax_, shape, margin[k:k + 1])[0][0] - depth[k]) / e
                                 for ax_ in np.eye(3)])
                axis = np.cross(pts[k] - head, -grad if np.linalg.norm(grad) > 1e-6 else out[k])

                if np.linalg.norm(axis) < 1e-9:
                    break
                # about as far as it takes to lift that point out (depth / lever arm), a step at a time
                step = np.clip(depth[k] / max(np.linalg.norm(pts[k] - head), 1e-3) * 1.2,
                               np.radians(1.0), np.radians(STYLE["clear_step_deg"]))
                turn = Rot.from_rotvec(_norm(axis) * step)
                u, f = turn.apply(u), turn.apply(f)
            return c0, off, q, curve, chord, u, f

        # the arms as the dancer moves them, then kept out of the body frame by frame; the corrected directions
        # are smoothed over time so the arm glides around the body instead of jumping out and back between
        # frames, and only what smoothing lets sink in again is corrected once more
        rest_dir = _norm(np.asarray(M[f"{name}__tail"], float) - np.asarray(M[f"{name}__head"], float))
        raw = [(match_rest(T[i].T @ upper[side_name][i], rest_dir),
                match_rest(T[i].T @ fore[side_name][max(0, i - lag_f)], rest_dir)) for i in range(n)]
        am = mv["arm"][side_name]  # the agent's own arm moves take over from the copy where they are set
        raw = [(_norm((1 - am["w"][i]) * u + am["w"][i] * am["u"][i]),
                _norm((1 - am["w"][i]) * f + am["w"][i] * am["f"][i])) if am["w"][i] > 0 else (u, f)
               for i, (u, f) in enumerate(raw)]
        if shape is not None:
            fixed = np.array([place_arm(u, f, float(knob["arms"][i]), float(am["w"][i]))[5:]
                              for i, (u, f) in enumerate(raw)])
            moved = np.degrees(np.arccos(np.clip((fixed[:, 0] * _norm(np.array([r[0] for r in raw]))).sum(1), -1, 1)))
            clear_report[side_name] = {"frames_turned": round(float((moved > 1).mean()), 3),
                                       "mean_deg": round(float(moved.mean()), 1),
                                       "p90_deg": round(float(np.percentile(moved, 90)), 1)}
            sig = STYLE["clear_smooth_s"] * fps
            fixed = _norm(np.stack([np.stack([smooth(fixed[:, a, k], sig) for k in range(3)], -1)
                                    for a in range(2)], 1))
            raw = [(fixed[i, 0], fixed[i, 1]) for i in range(n)]
        for i in range(n):
            c0, off, q, curve, chord, u, f = place_arm(*raw[i], float(knob["arms"][i]), float(am["w"][i]))
            if socket:
                ch[f"{socket}.quat"].append(_local(M[socket], ident, c0))
                ch[f"{socket}.loc"].append([round(float(x), 4) for x in M[socket].T @ off])  # -> bone rest frame
            ch[f"{name}.quat"].append(q)
            ch[f"{name}.bbone"].append(curve)
            ch[f"{name}.scale"].append([1.0, chord, 1.0])
            fist[i] = fist_amount(u, f)
            if not np.isnan(am["fist"][i]):
                fist[i] = am["fist"][i]
        # palm and back are equivalent, so keep the hand roll continuous (mod pi) instead of flipping
        bb = ch[f"{name}.bbone"]
        if bb:
            r = np.array([c[5] for c in bb], float)
            for i in range(1, n):
                r[i] -= np.pi * np.round((r[i] - r[i - 1]) / np.pi)
            # a big twist pinches a raised arm, so fade it out above the shoulder
            up = np.array([raise_deg(_norm(u)) for u, _ in raw])
            keep = np.clip((150.0 - up) / 50.0, 0.0, 1.0)
            r = smooth(np.clip(r, -np.radians(120), np.radians(120)) * smooth(keep, 0.1 * fps), 0.05 * fps)
            for i in range(n):
                bb[i][5] = round(float(r[i]), 5)
        if f"fist_{side_name}" in rig.get("shape_keys", {}):
            ch[f"{name}.fist"] = [round(float(x), 3) for x in np.clip(smooth(fist, 0.05 * fps), 0, 1)]

    return {"fps": fps, "n_frames": n, "character": rig.get("name", "character"), "channels": ch,
            "retarget": {"body": BODY, "soft": SOFT, "style": STYLE, "groove": GROOVE,
                         "arms_turned_out_of_body": clear_report},
            "leg_events": [{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in ev.items()}
                           for ev in legs["events"]]}
