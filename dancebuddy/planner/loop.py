"""The planner: the LLM at the centre of the DanceBuddy agent, and the loop it runs in.

The model reads the dance and the character through tools (dancebuddy/tools.py), submits plans (actions/), which
are carried out (executor/) and checked (critics/), looks where it chooses, and revises. The best attempt is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

from dancebuddy import llm
from dancebuddy.critics.numbers import rank
from dancebuddy.tools import Toolbox

SYSTEM = """You are DanceBuddy, an animation agent. A non-humanoid plush character copies a human dancer. A
deterministic copy already exists: the character's arms follow the dancer's arms, its body follows the dancer's
torso, every leg move is translated (the character has no legs) and the body grooves on the beat. You decide how
the dancer's moves are SHOWN on this body, unit by unit, by setting a plan.

How to work:
0. Only if you are asked for a NEW character: make it first with the character tools. Create it from what you
   were given (create_character_from_text / create_character_from_image); if it is a plush, blob or other soft body,
   segment_character it, then import_and_rig_soft the segmented GLB and check that its arms were found (else try
   `yaw` or another segmentation granularity). A humanoid or an animal with real legs can only be rig_humanoid'ed:
   say that the dance needs a soft-rigged body and stop. Each Tripo call takes minutes and costs credits: never
   repeat one that succeeded. The character you rig is the one you then dance with.
1. Read the dance (get_dance_score, get_dance_signals, get_leg_events) and the character (get_character_spec).
   Note what each unit is about: arms, footwork, head and hair flips, turns.
2. Read the plan format (get_plan_format). Every knob defaults to the plain copy.
3. For each unit decide what matters most and how this body can show it, following the character's spec:
   - moves the body can copy: keep them, exaggerate a little if the unit is about them (a small round character
     needs bigger moves to read as strongly as a person)
   - moves it cannot copy (head flips: it has no neck; footwork: it has no legs): move them onto parts it has
     (head -> upper body via the head knob, or emphasis nods/tilts; strong steps -> bigger leg translation)
   - accents: every accent should show as a clear hit; add an emphasis move where the copy has none
   - keep the character inside the picture and within its range
4. submit_plan. It returns checks per unit: missed accents, completeness (does the character move about
   as much as the dancer, relatively), frames out of the picture, range. Fix what the report points at and keep
   what works. Do not change units that pass unless you have a reason.
5. The numbers do not show what the character looks like. Check the latest attempt where it matters, cheapest
   first, and decide yourself where to look:
   - check_arm_geometry (a few seconds): arms folding or sinking into the body, on chosen frames or a unit
   - look_at_frames (about half a minute; limited number of looks): renders chosen frames small next to the
     dancer and returns a reviewer's notes per frame: does the character show the move, what looks wrong, a fix.
     Look at units with big arm moves, turns or accents, and at units the checks flagged.
   Turn what you learn into changes and submit again. Knobs scale whole units; for a particular moment write a
   move in the plan's `moves` (code lines, see get_plan_format): e.g. arm("both", 3.5, pose="overhead") when the
   hands should go above the head, arm("both", 0.7, pose="hands_together_chest"), body(2.7, tilt=-12). A move
   takes over from the copy around its time and blends back, so write moves only where the copy does not read,
   time them on the dancer's moment (seconds), and look again to confirm they worked.
6. When the checks pass and what you looked at is fine, or you are out of attempts, reply with a short summary of
   your decisions, what you checked and what is still wrong, and stop."""


def run_agent(kp: dict, score: dict, rig: dict, spec: dict, out_dir: str | Path, max_attempts: int = 4,
              max_turns: int = 24, model: str | None = None, blend: str | None = None,
              source: str | None = None, max_looks: int = 4, use_looks: bool = True, use_geometry: bool = True,
              use_moves: bool = True, new_character: str | None = None, character_request: str = "",
              tripo=None) -> dict:
    """Run the agent on one dance. blend: the rigged character (geometry check, looks); source: the dance video
    (looks). use_looks / use_geometry / use_moves switch a capability off (ablations): its tool is not offered and
    the agent is told. new_character: the agent first makes a character with this name (from character_request)
    and then dances with it; rig may be None. Writes trace.json, summary.json, the best plan / curves / checks and
    every attempt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    box = Toolbox(kp, score, rig, spec, out, blend=blend, source=source, model=model, max_attempts=max_attempts,
                  max_looks=max_looks, use_looks=use_looks, use_geometry=use_geometry, use_moves=use_moves,
                  new_character=new_character, tripo=tripo)
    missing = box.missing()
    note = f" In this run you do not have {', '.join(missing)}; skip those steps." if missing else ""
    ask = ("Make the character dance this dance. Start by reading the dance and the character." if not new_character
           else f"Make a new character ({character_request}) and then make it dance this dance. Start by making "
                "the character.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask + note}]
    final_text = ""
    calls_made, cost = 0, 0.0
    for _ in range(max_turns):
        resp = llm.call(messages, tools=box.specs(), model=model)
        calls_made += 1
        cost += float((resp.get("usage") or {}).get("cost") or 0.0)
        calls = resp["tool_calls"]
        assistant = {"role": "assistant", "content": resp["text"] or None}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in calls]
        messages.append(assistant)
        if not calls:
            final_text = resp["text"]
            break
        for c in calls:
            a = c["arguments"]
            result = (json.dumps({"error": "arguments were not valid JSON", "details": a["_parse_error"]})
                      if "_parse_error" in a else box.call(c["name"], a))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    (out / "trace.json").write_text(json.dumps(messages, indent=1))
    attempts, looks = box.attempts, box.looks
    if not attempts:
        raise RuntimeError("the agent never submitted a valid plan; see trace.json")
    best = max(attempts, key=lambda x: rank(x["result"]))
    for f in ("plan.json", "curves.json", "checks.json"):
        (out / f).write_text((Path(best["dir"]) / f).read_text())
    summary = {"attempts": len(attempts), "best_attempt": best["attempt"], "passed": best["result"]["passed"],
               "history": [{"attempt": x["attempt"], "passed": x["result"]["passed"],
                            "problems": sum(len(u["problems"]) for u in x["result"]["units"])} for x in attempts],
               "looks": [{"attempt": x["attempt"], "frames": x["frames"], "picture": x["picture"]} for x in looks],
               "final_message": final_text, "model": model or llm.DEFAULT_MODEL,
               "capabilities": box.use, "character": str(box.char_prefix) if box.char_prefix else None,
               "character_steps": box.charbox.made if box.charbox else [], "llm_calls": calls_made + len(looks),
               "cost_usd": round(cost + sum(x.get("cost_usd", 0.0) for x in looks), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"agent: {len(attempts)} attempt(s), best = {best['attempt']}, passed = {best['result']['passed']}")
    return summary
