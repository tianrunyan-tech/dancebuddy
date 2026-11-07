# DanceBuddy

**An agent that makes a non-humanoid character dance along with a human dancer.** CS329Z, Fall 2026.

Input: a video of one person dancing. Output: a video of a plush character (no legs, no neck, soft mitten arms)
dancing the same dance, side by side with the dancer.

How it works: MediaPipe extracts the dancer's pose; code turns it into a *dance score* (beats, accents, phrase units,
leg moves) and a deterministic copy of the motion onto the character's soft rig. An LLM agent (OpenRouter, tool
calling) reads the dance and the character through tools, submits plans that scale and accent the copy, gets numeric
checks back, inspects the arms and rendered frames with a vision model, and revises. It can also make a new character
first with Tripo3D (text or image to 3D, segmentation, rigging). Blender renders the result.

## Contents

1. [Quick start (about 10 minutes, no API key)](#1-quick-start)
2. [Setup](#2-setup)
3. [Run the agent on a dance](#3-run-the-agent-on-a-dance)
4. [Make a new character with the agent (Tripo3D)](#4-make-a-new-character-with-the-agent)
5. [Reproduce the evaluation](#5-reproduce-the-evaluation)
6. [Sample videos and characters](#6-sample-videos-and-characters)
7. [Repository layout](#7-repository-layout)

## 1. Quick start

This checks that everything is installed and renders the bundled plush dancing to a built-in synthetic dancer, without
any API key (the plan is the plain copy, no LLM):

```bash
git clone https://github.com/tianrunyan-tech/dancebuddy.git
cd dancebuddy
uv sync --extra render
uv run python -m dancebuddy check
uv run python -m evaluation prepare --dance synth
uv run python -m evaluation run --dance synth --systems plain_copy --no-llm
uv run python -m dancebuddy render runs/synth --systems plain_copy --preview
uv run python -m dancebuddy compose runs/synth --systems plain_copy --preview
```

Open `runs/synth/side_by_side_preview.mp4`. To see the agent itself, add an OpenRouter key (Section 2) and run
Section 3 on `runs/synth`.

## 2. Setup

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), FFmpeg (`ffmpeg` on your path), and Blender 5.x either as
the `bpy` Python module or as the Blender app.

```bash
uv sync --extra render        # installs bpy (about 400 MB); results were made with bpy 5.2.3
```

Or use an installed Blender app instead of `bpy`:

```bash
uv sync
export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
```

API keys go in a local `.env` (git-ignored):

```bash
cp .env.example .env
```

| Variable | Needed for | Where to get it |
|---|---|---|
| `OPENROUTER_API_KEY` | the agent and the LLM baseline | https://openrouter.ai/keys |
| `TRIPO_API_KEY` | only making new characters (Section 4) | https://platform.tripo3d.ai |
| `DANCEBUDDY_MODEL` (optional) | another model; default `openai/gpt-6-sol` | any OpenRouter model with tool calling and image input |

Check the setup and run the offline tests (synthetic data and mocked APIs, no keys needed):

```bash
uv run python -m dancebuddy check
uv run pytest
```

Every LLM request is cached in `.cache/llm/`; tokens and cost per call are logged in `.cache/llm/ledger.jsonl`, and
each run writes its own cost to `summary.json`. A run costs about $0.10.

## 3. Run the agent on a dance

Use a 10–30 s video of one person, whole body in view, filmed from the front. Use a new run folder for each video.

```bash
uv run python -m dancebuddy perceive path/to/dance.mp4 runs/my_dance
uv run python -m dancebuddy agent runs/my_dance --no-moves
uv run python -m dancebuddy render runs/my_dance --engine eevee
uv run python -m dancebuddy compose runs/my_dance
```

The result is `runs/my_dance/side_by_side.mp4`. For a quick untextured check, add `--preview` to `render` and
`compose`; `--engine cycles` (the default) is slower and higher quality.

Useful options for `agent`:

- `--char assets/<name>_soft` dances another rigged character (default `assets/plush_soft`); pass the same `--char`
  to `render`.
- `--name <folder>` writes the run to `runs/my_dance/<folder>` (default `agent`); pass `--systems <folder>` to
  `render` and `compose`.
- `--no-moves`, `--no-looks`, `--no-geometry` turn one capability off (ablations). We recommend `--no-moves` for
  now: the agent's own moves can blend in within a few frames and look like snapping.
- `--attempts`, `--looks` set the budgets (default 4 each).

What a run writes, in `runs/my_dance/<name>/`:

| File | Content |
|---|---|
| `trace.json` | the whole conversation: every message, tool call and tool result |
| `attempt_N/` | each submitted plan, its motion curves and its checks |
| `plan.json`, `curves.json`, `checks.json` | the best attempt |
| `summary.json` | attempts, checks history, looks, final message, LLM calls and cost |

`python -m dancebuddy replay runs/my_dance PLAN.json --name X` carries out a saved plan again without the LLM.

## 4. Make a new character with the agent

With a Tripo3D key, the same agent can make the character first and then dance with it:

```bash
uv run python -m dancebuddy agent runs/my_dance --new-character blob --image path/to/picture.png
uv run python -m dancebuddy agent runs/my_dance --new-character blob --prompt "a plush toy with blue mitten arms"
uv run python -m dancebuddy agent runs/my_dance --new-character blob --views front=f.png left=l.png back=b.png
```

The agent creates the model, segments it into parts (so the arms come loose from the body), imports it and builds the
soft rig as `assets/blob_soft`, then plans the dance. Render with `--char assets/blob_soft`. The name must not contain
`_`. Each Tripo job takes a few minutes and uses credits.

You can also import a GLB you already have, without Tripo:

```bash
uv run python -m dancebuddy character model path/to/model.glb assets/blob     # add --yaw 90 if it faces sideways
cp assets/plush_motion_spec.json assets/blob_motion_spec.json
uv run python -m dancebuddy character rig assets/blob assets/blob_soft
```

The rig finds the arms by their colour (blue) and the hair (dark brown) from the textures, so it fits plush-like
characters; check `assets/blob_views/front.png` after importing.

## 5. Reproduce the evaluation

Systems: `pose_copy`, `plain_copy`, `planner` (baselines), `agent`, `agent_no_looks`, `agent_no_moves`,
`agent_numbers` (ablations). Metrics: accent recall, beat precision, motion correlation, arm follow, hands in frame,
arm folds and sinking, rhythm measured on rendered pixels, and cost (see `evaluation/metrics/__init__.py`).

1. Put the dance videos (Section 6) at the paths listed in `evaluation/data/dances.json`.
2. Run:

```bash
uv run python -m evaluation prepare              # perception on every dance
uv run python -m evaluation run                  # all systems; --systems ..., --dance <id>, --no-llm, --force
uv run python -m evaluation metrics --arms       # -> evaluation/results/results.{json,md}
```

LLM systems are nondeterministic, so numbers are close to, not identical with, reported ones. To re-score a set of
saved runs without keys or videos:

```bash
uv run python -m evaluation export --name mine   # saves runs (no videos) to evaluation/results/mine/
uv run python -m evaluation metrics --runs evaluation/results/mine/runs --arms --out evaluation/results/mine/recomputed.json
```

The fixed-rule baseline from the midway report (plain copy plus a full squash on every accent) is a plan you can
replay without the LLM:

```bash
uv run python -c "import json; from dancebuddy.actions.plan import default_plan; s=json.load(open('runs/my_dance/dance_score.json')); p=default_plan(s); p['emphasis']=[{'frame':a,'move':'squash','size':1.0} for a in s['accents']]; json.dump(p, open('squash_rule.json','w'))"
uv run python -m dancebuddy replay runs/my_dance squash_rule.json --name squash_rule
```

## 6. Sample videos and characters

Dance videos and character models are not committed (size and licensing). Download them from
**[Google Drive: DanceBuddy samples](LINK_TO_BE_ADDED)** and place them as follows:

| File | Put it at | What it is |
|---|---|---|
| `dance.mp4`, `dance2.mp4` | `videos/` | the two dance clips used in the midway report |
| `plush_unrigged.glb` | anywhere | our plush model, unrigged |
| `latest.glb` | anywhere | the plush segmented into parts with Tripo3D |

Then, for example: `uv run python -m dancebuddy perceive videos/dance2.mp4 runs/dance2`, and import a GLB as in
Section 4.

## 7. Repository layout

```
dancebuddy/
  perception/   video.py (MediaPipe pose), beats.py, signals.py     -> keypoints + dance score
  planner/      loop.py: the agent loop (system prompt, tool calling, best attempt)
  tools.py      the agent's tools: specs (JSON schema) + handlers, one Toolbox per run
  actions/      plan.py (knobs, legs, emphasis), moves.py (the agent's own moves)
  executor/     retarget.py (deterministic copy + plan -> curves), legs.py, render.py
  critics/      numbers.py (numeric checks), inspect.py (arm geometry, vision review)
  characters/   tripo.py (Tripo3D client), make.py (character tools), puppet.py
  knowledge/    character rig / motion spec / card loading
  blender/      scripts run inside Blender: make_model, rig_soft, render_curves, ...
  llm.py        OpenRouter wrapper with cache and cost ledger
evaluation/     baselines, data lists, metrics, results
assets/         rigged characters, motion specs, rendered views
tests/          offline tests
```
