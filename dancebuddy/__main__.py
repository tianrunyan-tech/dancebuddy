"""DanceBuddy: an agent that makes a non-humanoid character dance along with a human dancer.

  python -m dancebuddy check                          # is everything installed and configured?

Per dance:
  python -m dancebuddy perceive VIDEO runs/my          # perception: pose -> keypoints + dance score
  python -m dancebuddy agent    runs/my                # the agent: read, plan, act, check, look, revise
  python -m dancebuddy agent    runs/my --new-character blob --image PIC   # ...after making the character (Tripo)
  python -m dancebuddy render   runs/my                # executor: render the agent's result (Cycles, 30 fps)
  python -m dancebuddy compose  runs/my                # dancer | agent side by side
  python -m dancebuddy demo     VIDEO runs/my          # all of the above, in order
  python -m dancebuddy replay   runs/my PLAN.json      # carry out a saved plan again, without the LLM

Characters (once each):
  python -m dancebuddy character model   MODEL.glb assets/plush        # import an unrigged 3D model
  python -m dancebuddy character rig     assets/plush assets/plush_soft  # rig it from its motion spec
  python -m dancebuddy character capsule assets/capsule                # the built-in 3D capsule
  python -m dancebuddy character puppet  PICTURE assets/blob           # a picture as a rigged 2D cut-out
  python -m dancebuddy character card    assets/plush_soft             # describe its body (vision model)

Evaluation (baselines, data, metrics) is a separate module: python -m evaluation --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dancebuddy.executor.render import blender_cmd, compose, render_system, run
from dancebuddy.knowledge import character

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAR = ROOT / "assets" / "plush_soft"


# ------------------------------------------------------------------ set-up check

def cmd_check(a):
    """Check what the project needs: Python packages, Blender, ffmpeg, the characters, the OpenRouter key."""
    import importlib.util
    import os
    import shutil
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(("  ok   " if good else "  MISSING ") + what + ("" if good or not fix else f"\n         -> {fix}"))

    print("DanceBuddy set-up check")
    for mod in ("numpy", "scipy", "cv2", "mediapipe", "openai", "jsonschema"):
        line(importlib.util.find_spec(mod) is not None, f"python package {mod}", "run: uv sync --extra render")
    bpy = importlib.util.find_spec("bpy") is not None
    line(bpy or os.environ.get("BLENDER") or shutil.which("blender"),
         "Blender (" + ("bpy module" if bpy else os.environ.get("BLENDER") or shutil.which("blender") or "none") + ")",
         "run: uv sync --extra render   (or install Blender 4.2+ and export BLENDER=/path/to/blender)")
    line(shutil.which("ffmpeg"), "ffmpeg (for side-by-side videos)", "install ffmpeg (brew / apt / winget)")
    line(Path(str(DEFAULT_CHAR) + ".blend").exists(), f"character {DEFAULT_CHAR.relative_to(ROOT)}",
         "the .blend files are in the repository; check that the clone is complete (git lfs is not used)")
    from dancebuddy import llm  # loads .env
    key = os.environ.get("OPENROUTER_API_KEY", "")
    line(key and "..." not in key, "OPENROUTER_API_KEY (needed by the agent and the LLM baselines only)",
         "cp .env.example .env and put your key in it (https://openrouter.ai/keys)")
    if a.llm and key:
        try:
            r = llm.call([{"role": "user", "content": "Reply with the word ok."}], max_tokens=20)
            line("ok" in (r["text"] or "").lower(), f"a call to {llm.DEFAULT_MODEL}")
        except Exception as e:  # noqa: BLE001
            line(False, f"a call to {llm.DEFAULT_MODEL}: {e}")
    print("all set" if ok else "fix the MISSING items above (the tests and the no-LLM steps work without a key)")


# ------------------------------------------------------------------ per dance

def cmd_perceive(a):
    from dancebuddy.perception.video import perceive
    Path(a.run).mkdir(parents=True, exist_ok=True)
    perceive(a.video, a.run, a.max_seconds)
    if Path(a.video).exists():
        (Path(a.run) / "source.txt").write_text(str(Path(a.video).resolve()))


def cmd_agent(a):
    from dancebuddy.perception.video import load_keypoints
    from dancebuddy.planner.loop import run_agent
    run_dir = Path(a.run)
    score = json.loads((run_dir / "dance_score.json").read_text())
    source = (run_dir / "source.txt").read_text().strip() if (run_dir / "source.txt").exists() else score.get("source")
    request, new = "", a.new_character
    if new:
        if "_" in new:
            raise SystemExit("--new-character name must not contain '_'")
        if a.image:
            request = f"from the picture {Path(a.image).expanduser().resolve()}"
        elif a.views:
            request = "from these views: " + ", ".join(
                f"{v.split('=', 1)[0]}={Path(v.split('=', 1)[1]).expanduser().resolve()}" for v in a.views)
        elif a.prompt:
            request = f"described as: {a.prompt}"
        else:
            raise SystemExit("--new-character needs --prompt, --image or --views")
    rig = None if new else character.rig(a.char)
    run_agent(load_keypoints(run_dir / "keypoints.npz"), score, rig, None if new else character.spec(a.char),
              run_dir / a.name, max_attempts=a.attempts, model=a.model,
              blend=None if new else character.blend(a.char),
              source=source if source and Path(source).exists() else None, max_looks=a.looks,
              use_looks=not a.no_looks, use_geometry=not a.no_geometry, use_moves=not a.no_moves,
              new_character=new, character_request=request)


def cmd_replay(a):
    """Carry out a saved plan (e.g. an agent's plan.json) again, without the LLM, and check it."""
    from dancebuddy.critics.numbers import check_curves
    from dancebuddy.executor.retarget import retarget
    from dancebuddy.perception.signals import compute_signals
    from dancebuddy.perception.video import load_keypoints
    run_dir = Path(a.run)
    score = json.loads((run_dir / "dance_score.json").read_text())
    kp, rig = load_keypoints(run_dir / "keypoints.npz"), character.rig(a.char)
    curves = retarget(kp, rig, signals=compute_signals(kp), beats=score["beats"],
                      plan=json.loads(Path(a.plan).read_text()), score=score)
    out = run_dir / a.name
    out.mkdir(exist_ok=True)
    (out / "curves.json").write_text(json.dumps(curves))
    (out / "checks.json").write_text(json.dumps(check_curves(curves, score, rig, character.spec(a.char)), indent=1))
    print(f"replay: {curves['n_frames']} frames -> {out}")


def cmd_render(a):
    for s in a.systems:
        render_system(Path(a.run), s, a.char, a.engine, a.width, a.height, a.fps, a.preview, a.max_frames)


def cmd_compose(a):
    compose(Path(a.run), a.systems, a.preview, a.height)


def cmd_demo(a):
    cmd_perceive(argparse.Namespace(video=a.video, run=a.run, max_seconds=a.max_seconds))
    cmd_agent(argparse.Namespace(run=a.run, char=a.char, name="agent", attempts=a.attempts, looks=a.looks, model=None,
                                 no_looks=False, no_geometry=False, no_moves=False))
    render_system(Path(a.run), "agent", a.char, a.engine, a.width, a.height, 30, a.preview)
    compose(Path(a.run), ["agent"], a.preview, a.height)


# ------------------------------------------------------------------ characters

def cmd_char_model(a):
    args = ["--model", str(Path(a.model).expanduser().resolve()), "--out", str(a.out), "--engine", a.engine]
    if a.yaw:
        args += ["--yaw", str(a.yaw)]
    run(blender_cmd("make_model.py", args))


def cmd_char_rig(a):
    run(blender_cmd("rig_soft.py", ["--blend", str(a.model) + ".blend", "--out", str(a.out)]))


def cmd_char_capsule(a):
    run(blender_cmd("make_capsule.py", ["--out", str(a.out), "--engine", a.engine]))


def cmd_char_puppet(a):
    from dancebuddy.characters.puppet import cut_out
    from dancebuddy.knowledge.card import build_card
    cut_out(a.image, a.out)
    run(blender_cmd("make_puppet.py", ["--out", str(a.out), "--engine", a.engine]))
    build_card(Path(str(a.out) + "_rig.json"), character.card_path(Path(a.out)), use_vlm=not a.no_vlm)


def cmd_char_card(a):
    from dancebuddy.knowledge.card import build_card
    card = build_card(Path(str(a.char) + "_rig.json"), character.card_path(Path(a.char)), use_vlm=not a.no_vlm)
    print(json.dumps(card.get("description"), indent=1))


# ------------------------------------------------------------------ command line

def main(argv=None):
    p = argparse.ArgumentParser(prog="dancebuddy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(parent, name, fn, *args):
        s = parent.add_parser(name)
        s.set_defaults(fn=fn)
        for arg in args:
            s.add_argument(arg)
        return s

    def char_opt(s):
        s.add_argument("--char", type=Path, default=DEFAULT_CHAR, help="rigged character prefix (default assets/plush_soft)")

    def render_opts(s):
        s.add_argument("--engine", default="cycles", choices=["auto", "eevee", "cycles", "workbench"])
        s.add_argument("--width", type=int, default=540)
        s.add_argument("--height", type=int, default=960)

    def agent_opts(s):
        s.add_argument("--attempts", type=int, default=4, help="plans the agent may submit")
        s.add_argument("--looks", type=int, default=4, help="times the agent may look at rendered frames")

    s = add(sub, "check", cmd_check); s.add_argument("--llm", action="store_true", help="also make one small LLM call")
    s = add(sub, "perceive", cmd_perceive, "video", "run"); s.add_argument("--max-seconds", type=float)
    s = add(sub, "agent", cmd_agent, "run"); char_opt(s); agent_opts(s)
    s.add_argument("--name", default="agent", help="output folder in the run (default agent)")
    s.add_argument("--model", help="LLM (OpenRouter id; default $DANCEBUDDY_MODEL or openai/gpt-6-sol)")
    s.add_argument("--no-looks", action="store_true", help="ablation: no looking at rendered frames")
    s.add_argument("--no-geometry", action="store_true", help="ablation: no arm geometry check")
    s.add_argument("--no-moves", action="store_true", help="ablation: knobs only, no own moves")
    s.add_argument("--new-character", metavar="NAME", help="first make a new character (Tripo) and dance with it")
    s.add_argument("--prompt", help="new character: a description"); s.add_argument("--image", help="new character: a picture")
    s.add_argument("--views", nargs="+", help="new character: side=path, e.g. front=f.png left=l.png")
    s = add(sub, "replay", cmd_replay, "run", "plan"); char_opt(s); s.add_argument("--name", default="replay")
    s = add(sub, "render", cmd_render, "run"); char_opt(s); render_opts(s)
    s.add_argument("--systems", nargs="+", default=["agent"]); s.add_argument("--max-frames", type=int, default=0)
    s.add_argument("--fps", type=float, default=30, help="render frame rate (0 = the clip's own; default 30)")
    s.add_argument("--preview", action="store_true", help="quick untextured preview into <system>/preview")
    s = add(sub, "compose", cmd_compose, "run"); s.add_argument("--systems", nargs="+", default=["agent"])
    s.add_argument("--preview", action="store_true"); s.add_argument("--height", type=int, default=960)
    s = add(sub, "demo", cmd_demo, "video", "run"); char_opt(s); render_opts(s); agent_opts(s)
    s.add_argument("--max-seconds", type=float); s.add_argument("--preview", action="store_true")

    ch = sub.add_parser("character", help="make a character").add_subparsers(dest="what", required=True)
    s = add(ch, "model", cmd_char_model, "model", "out"); s.add_argument("--engine", default="auto")
    s.add_argument("--yaw", type=float, default=0.0, help="rotate the model (degrees) if it does not face the camera")
    s = add(ch, "rig", cmd_char_rig, "model", "out")
    s = add(ch, "capsule", cmd_char_capsule, "out"); s.add_argument("--engine", default="auto")
    s = add(ch, "puppet", cmd_char_puppet, "image", "out"); s.add_argument("--engine", default="auto")
    s.add_argument("--no-vlm", action="store_true")
    s = add(ch, "card", cmd_char_card, "char"); s.add_argument("--no-vlm", action="store_true")

    a = p.parse_args(argv)
    if hasattr(a, "char") and a.char is not None and not isinstance(a.char, Path):
        a.char = Path(a.char)
    a.fn(a)


if __name__ == "__main__":
    main()
