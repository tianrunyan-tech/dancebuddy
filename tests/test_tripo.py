"""Tripo client and the agent's character tools, against a mocked Tripo API (no key, no network)."""

import json

import httpx
import pytest

from dancebuddy.characters.make import CHARACTER_SPECS, CharacterTools
from dancebuddy.characters.tripo import Tripo, TripoError

TASK = "123e4567-e89b-42d3-a456-426614174000"


def fake_api(seen, fail=False):
    polls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.content))
        if req.url.path == "/v3/files":
            return httpx.Response(200, json={"code": 0, "data": {"file_token": "file_abc"}})
        if req.method == "POST":
            return httpx.Response(200, json={"code": 0, "data": {"task_id": TASK}})
        if req.url.path == f"/v3/tasks/{TASK}":
            polls["n"] += 1
            if polls["n"] < 2:
                return httpx.Response(200, json={"code": 0, "data": {"task_id": TASK, "status": "running"}})
            if fail:
                return httpx.Response(200, json={"code": 0, "data": {"task_id": TASK, "status": "failed",
                                                                        "error_message": "bad input"}})
            return httpx.Response(200, json={"code": 0, "data": {
                "task_id": TASK, "status": "success", "credits_consumed": 30,
                "output": {"model_url": "https://cdn.tripo3d.ai/out.glb"}}})
        if req.url.host == "cdn.tripo3d.ai":
            return httpx.Response(200, content=b"glTF-binary")
        return httpx.Response(404, json={"code": 1, "message": "not found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_segment_runs_submit_poll_download(tmp_path):
    seen = []
    t = Tripo("k", client=fake_api(seen), poll_s=0)
    res = t.segment("task:" + TASK, tmp_path / "seg.glb")
    assert (tmp_path / "seg.glb").read_bytes() == b"glTF-binary"
    assert res["task_id"] == TASK and res["credits"] == 30
    body = json.loads(seen[0][2])
    assert seen[0][1] == "/v3/mesh/segment" and body["input"] == TASK


def test_local_image_is_uploaded(tmp_path):
    seen = []
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    Tripo("k", client=fake_api(seen), poll_s=0).image_to_model(str(img), tmp_path / "m.glb")
    assert seen[0][1] == "/v3/files"
    assert json.loads(seen[1][2])["input"] == "file_abc"


def test_rig_humanoid_uses_biped_mixamo(tmp_path):
    seen = []
    Tripo("k", client=fake_api(seen), poll_s=0).rig("https://x/m.glb", tmp_path / "r.glb")
    body = json.loads(seen[0][2])
    assert body["rig_type"] == "biped" and body["spec"] == "mixamo"


def test_failed_task_is_reported_to_the_agent(tmp_path):
    box = CharacterTools("blob", tmp_path, Tripo("k", client=fake_api([], fail=True), poll_s=0))
    out = json.loads(box.call("create_character_from_text", {"prompt": "a blob"}))
    assert "bad input" in out["error"]


def test_tools_are_registered():
    assert set(CHARACTER_SPECS) == {"create_character_from_text", "create_character_from_image", "segment_character",
                          "rig_humanoid", "import_and_rig_soft"}
    with pytest.raises(TripoError):
        Tripo("k", client=fake_api([]), poll_s=0).as_input("not-a-file")


def test_one_agent_makes_the_character_then_dances(tmp_path, monkeypatch):
    """With --new-character the dance agent's toolbox offers the character tools too; dance tools wait for the
    character, and a successful import_and_rig_soft makes it the character the agent dances with."""
    from dancebuddy import tools
    from evaluation.data.synth import make
    from dancebuddy.perception.video import build_dance_score, load_keypoints

    d = make(tmp_path / "synth")
    kp = load_keypoints(d / "keypoints.npz")
    score = build_dance_score(kp)
    box = tools.Toolbox(kp, score, None, None, tmp_path / "run", new_character="blob",
                        tripo=Tripo("k", client=fake_api([]), poll_s=0))
    names = [s["function"]["name"] for s in box.specs()]
    assert "segment_character" in names and "submit_plan" in names
    assert "no character yet" in json.loads(box.call("get_character_spec", {}))["error"]

    plush = tools.ROOT / "assets" / "plush_soft"
    monkeypatch.setattr(CharacterTools, "t_import_and_rig_soft",
                        lambda self, a: {"character": "assets/plush_soft", "arms_found": True})
    res = json.loads(box.call("import_and_rig_soft", {"model": "x.glb"}))
    assert "now the character you dance with" in res["note"] and box.char_prefix == plush
    assert "body" in json.dumps(json.loads(box.call("get_character_spec", {})))
