"""What the agent knows about a character, by its prefix (e.g. assets/plush_soft):

  <prefix>.blend              the rigged character (for rendering and the geometry / vision checks)
  <prefix>_rig.json           its bones, rest geometry, body shape and arm sizes (rig_soft.py)
  <name>_motion_spec.json     how it moves: qualitative rules learned from reference clips (name = prefix up to "_")
  <prefix>_card.json          a short description of its body (card.py), written on first use if missing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rig(char: Path) -> dict:
    p = Path(str(char) + "_rig.json")
    if not p.exists():
        sys.exit(f"no rigged character at {char} (missing {p.name})")
    return json.loads(p.read_text())


def spec(char: Path) -> dict:
    char = Path(char)
    p = char.parent / (char.name.split("_")[0] + "_motion_spec.json")
    return json.loads(p.read_text()) if p.exists() else {}


def blend(char: Path) -> str | None:
    p = Path(str(char) + ".blend")
    return str(p) if p.exists() else None


def card_path(char: Path) -> Path:
    return Path(str(char) + "_card.json")


def card(char: Path) -> dict:
    p = card_path(char)
    if p.exists():
        return json.loads(p.read_text())
    from dancebuddy.knowledge.card import build_card
    return build_card(Path(str(char) + "_rig.json"), p, use_vlm=False)
