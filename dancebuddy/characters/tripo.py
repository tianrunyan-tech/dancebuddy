"""Thin client for the Tripo3D v3 API (https://developers.tripo3d.ai): make, split and rig 3D characters.

Every Tripo job is a task: POST creates it and returns {"code": 0, "data": {"task_id": ...}}, then
GET /v3/tasks/{task_id} is polled until status is success / failed / cancelled. A successful task's
output.model_url is a GLB that expires after about 5 minutes, so it is downloaded at once.

An `input` (image, model) can be given as a local file (uploaded with POST /v3/files -> file_token), an
https URL, or the task_id of an earlier Tripo task ("task:<id>" or a bare uuid).

Set TRIPO_API_KEY (in .env or your shell).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = "https://openapi.tripo3d.ai/v3"
GEN_MODEL = "v3.1-20260211"          # text/image/multiview -> model
RIG_MODEL_HUMANOID = "v3.0-20260909"  # biped (humanoid) rigs
RIG_MODEL_OTHER = "v2.5-20260210"     # non-humanoid rigs
SEGMENT_MODEL = "v2.0-20260430"       # parts, with granularity
RIG_TYPES = ["biped", "quadruped", "hexapod", "octopod", "avian", "serpentine", "aquatic", "others"]
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class TripoError(RuntimeError):
    pass


class Tripo:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None, poll_s: float = 2.0,
                 timeout_s: float = 900.0):
        self.key = api_key or os.environ.get("TRIPO_API_KEY")
        if not self.key:
            raise TripoError("TRIPO_API_KEY is not set (add it to .env)")
        self.http = client or httpx.Client(timeout=120)
        self.poll_s, self.timeout_s = poll_s, timeout_s

    # ---- low level
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def _data(self, r: httpx.Response) -> dict:
        body = r.json()
        if r.status_code >= 400 or body.get("code", 0) != 0:
            raise TripoError(f"{body.get('message', r.text)} ({body.get('suggestion', '')})".strip())
        return body["data"]

    def upload(self, path: str | Path) -> str:
        """Local image or model -> file_token."""
        p = Path(path)
        with p.open("rb") as f:
            r = self.http.post(f"{BASE}/files", headers=self._headers(), files={"file": (p.name, f)})
        return self._data(r)["file_token"]

    def as_input(self, x: str) -> str:
        """A local path (uploaded), an https URL, or an earlier task id ('task:<id>' or a bare uuid)."""
        if x.startswith("task:"):
            return x[5:]
        if x.startswith(("http://", "https://")) or UUID.match(x):
            return x
        if Path(x).expanduser().exists():
            return self.upload(Path(x).expanduser())
        raise TripoError(f"input {x!r} is not a file, URL or task id")

    def submit(self, endpoint: str, payload: dict) -> str:
        r = self.http.post(f"{BASE}/{endpoint}", headers={**self._headers(), "Content-Type": "application/json"},
                           json=payload)
        return self._data(r)["task_id"]

    def wait(self, task_id: str) -> dict:
        t0 = time.time()
        while True:
            task = self._data(self.http.get(f"{BASE}/tasks/{task_id}", headers=self._headers()))
            status = task.get("status")
            if status == "success":
                return task
            if status in ("failed", "cancelled", "banned", "expired"):
                raise TripoError(f"task {task_id} {status}: {task.get('error_message') or task.get('error_code')}")
            if time.time() - t0 > self.timeout_s:
                raise TripoError(f"task {task_id} still {status} after {self.timeout_s:.0f} s")
            time.sleep(self.poll_s)

    def download(self, task: dict, dest: str | Path) -> Path:
        url = (task.get("output") or {}).get("model_url")
        if not url:
            raise TripoError(f"task {task.get('task_id')} has no model_url in its output")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = self.http.get(url, follow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest

    def run(self, endpoint: str, payload: dict, dest: str | Path) -> dict:
        """Submit, wait, download the GLB. Returns what a tool reports back."""
        task_id = self.submit(endpoint, payload)
        task = self.wait(task_id)
        path = self.download(task, dest)
        return {"task_id": task_id, "glb": str(path), "credits": task.get("credits_consumed"),
                "preview": (task.get("output") or {}).get("rendered_image_url")}

    # ---- the four jobs
    def text_to_model(self, prompt: str, dest: str | Path) -> dict:
        return self.run("generation/text-to-model", {"prompt": prompt, "model": GEN_MODEL, "texture": True,
                                                     "pbr": True, "texture_quality": "detailed"}, dest)

    def image_to_model(self, image: str, dest: str | Path) -> dict:
        return self.run("generation/image-to-model", {"input": self.as_input(image), "model": GEN_MODEL,
                                                      "texture": True, "pbr": True,
                                                      "texture_quality": "detailed"}, dest)

    def multiview_to_model(self, views: dict[str, str], dest: str | Path) -> dict:
        """views: {"front": ..., "left": ..., "back": ..., "right": ...} (front required)."""
        inputs = [{side: self.as_input(views[side])} for side in ("front", "left", "back", "right") if side in views]
        return self.run("generation/multiview-to-model", {"inputs": inputs, "model": GEN_MODEL, "texture": True,
                                                          "pbr": True}, dest)

    def segment(self, model: str, dest: str | Path, granularity: str = "balanced") -> dict:
        return self.run("mesh/segment", {"input": self.as_input(model), "model": SEGMENT_MODEL,
                                         "segmentation_granularity": granularity}, dest)

    def rig(self, model: str, dest: str | Path, rig_type: str = "biped", spec: str = "mixamo") -> dict:
        if rig_type not in RIG_TYPES:
            raise TripoError(f"rig_type must be one of {RIG_TYPES}")
        payload = {"input": self.as_input(model), "rig_type": rig_type, "out_format": "glb",
                   "model": RIG_MODEL_HUMANOID if rig_type == "biped" else RIG_MODEL_OTHER}
        if rig_type == "biped":
            payload["spec"] = spec
        return self.run("animations/rig", payload, dest)
