"""Character tools of the DanceBuddy agent: make a new dance-ready character with Tripo, then import and rig it.

The agent (planner/loop.py, through tools.py) calls these first when it is asked for a new character, then
dances with the result. Tripo jobs take minutes and cost credits, so they are offered only in such a run.

  tools  create_character_from_text   Tripo text-to-model
         create_character_from_image  Tripo image-to-model (one picture) or multiview-to-model (front/left/back/right)
         segment_character            Tripo mesh/segment: split the model into parts (arms come loose from the body)
         rig_humanoid                 Tripo animations/rig: a skeleton for humanoid or animal bodies (not our soft rig)
         import_and_rig_soft          local: import the GLB (make_model.py) and build the soft rig (rig_soft.py)

  python -m dancebuddy agent runs/my --new-character blob --image picture.png
  python -m dancebuddy agent runs/my --new-character blob --prompt "a plush toy with blue mitten arms"
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dancebuddy import llm
from dancebuddy.characters.tripo import RIG_TYPES, Tripo, TripoError
from dancebuddy.executor.render import blender_cmd, run

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
TEMPLATE_SPEC = ASSETS / "plush_motion_spec.json"

INPUT = {"type": "string", "description": "a local file path, an https URL, or 'task:<id>' of an earlier Tripo task"}
CHARACTER_SPECS = {
    "create_character_from_text": llm.tool_spec(
        "create_character_from_text", "Generate a textured 3D character from a text description (Tripo, about 1-3 "
        "minutes). Describe a front-facing character standing upright, arms a little away from the body. Returns the "
        "GLB path and its task id.",
        {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}),
    "create_character_from_image": llm.tool_spec(
        "create_character_from_image", "Generate a textured 3D character from one picture (`image`) or from views "
        "of it (`views`: front required, left/back/right optional). Tripo, about 1-3 minutes.",
        {"type": "object", "properties": {
            "image": INPUT,
            "views": {"type": "object", "properties": {k: INPUT for k in ("front", "left", "back", "right")},
                      "required": ["front"]}}}),
    "segment_character": llm.tool_spec(
        "segment_character", "Split a 3D model into parts (Tripo mesh/segment), so arms, hair and body are separate "
        "pieces instead of one fused surface. Do this before rigging a soft character: fused arms drag the body's "
        "skin when they rise.",
        {"type": "object", "properties": {
            "model": INPUT,
            "granularity": {"type": "string", "enum": ["simple", "balanced", "detailed"]}}, "required": ["model"]}),
    "rig_humanoid": llm.tool_spec(
        "rig_humanoid", "Give a model a full skeleton (Tripo animations/rig). Only for bodies with real limbs: a "
        "humanoid (biped, Mixamo bones) or an animal type. A plush or blob with soft arms and no legs should use "
        "import_and_rig_soft instead.",
        {"type": "object", "properties": {"model": INPUT, "rig_type": {"type": "string", "enum": RIG_TYPES}},
         "required": ["model"]}),
    "import_and_rig_soft": llm.tool_spec(
        "import_and_rig_soft", "Import a GLB into the project and build the soft rig the dance agent uses (spine, "
        "sliding shoulders, one bendable bone per arm). Returns how many vertices were found as arms, hair and body: "
        "a few hundred arm vertices or fewer means the arms were not found. yaw (degrees) turns a model that does "
        "not face the camera.",
        {"type": "object", "properties": {"model": {"type": "string", "description": "local GLB path"},
                                          "yaw": {"type": "number"}}, "required": ["model"]}),
}

class CharacterTools:
    def __init__(self, name: str, out: Path, tripo: Tripo | None = None):
        self.name, self.out = name, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tripo = tripo
        self.made: list[dict] = []

    def specs(self) -> list[dict]:
        return list(CHARACTER_SPECS.values())

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            res = fn(args)
        except (TripoError, FileNotFoundError, ValueError) as e:
            res = {"error": str(e)}
        self.made.append({"tool": name, "args": args, "result": res})
        return json.dumps(res)

    def _t(self) -> Tripo:
        if self.tripo is None:
            self.tripo = Tripo()
        return self.tripo

    def _dest(self, step: str) -> Path:
        return self.out / f"{self.name}_{step}.glb"

    # ---- Tripo
    def t_create_character_from_text(self, a):
        return self._t().text_to_model(a["prompt"], self._dest("text"))

    def t_create_character_from_image(self, a):
        if a.get("views"):
            return self._t().multiview_to_model(a["views"], self._dest("multiview"))
        if a.get("image"):
            return self._t().image_to_model(a["image"], self._dest("image"))
        return {"error": "give `image` or `views`"}

    def t_segment_character(self, a):
        return self._t().segment(a["model"], self._dest("segmented"), a.get("granularity", "balanced"))

    def t_rig_humanoid(self, a):
        return self._t().rig(a["model"], self._dest("rigged"), a.get("rig_type", "biped"))

    # ---- local
    def t_import_and_rig_soft(self, a):
        model = Path(a["model"]).expanduser().resolve()
        if not model.exists():
            raise FileNotFoundError(f"no model at {model}")
        prefix = ASSETS / self.name
        args = ["--model", str(model), "--out", str(prefix)]
        if a.get("yaw"):
            args += ["--yaw", str(a["yaw"])]
        run(blender_cmd("make_model.py", args), quiet=True)
        spec = ASSETS / f"{self.name}_motion_spec.json"
        if not spec.exists():
            shutil.copy(TEMPLATE_SPEC, spec)
        soft = ASSETS / f"{self.name}_soft"
        run(blender_cmd("rig_soft.py", ["--blend", str(prefix) + ".blend", "--out", str(soft)]), quiet=True)
        rig = json.loads(Path(str(soft) + "_rig.json").read_text())
        return {"character": str(soft.relative_to(ROOT)), "parts": rig.get("parts"),
                "views": str(Path(str(prefix) + "_views").relative_to(ROOT)),
                "arms_found": (rig.get("parts") or {}).get("arms", 0) > 500}
