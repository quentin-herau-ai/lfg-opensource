from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "checkpoints" / "pzow1k_seg_motion.pt",
    REPO_ROOT / "checkpoints" / "lfg.pt",
)
WEB_ROOT = REPO_ROOT / "outputs" / "web"
UPLOAD_ROOT = WEB_ROOT / "uploads"
RUN_ROOT = WEB_ROOT / "runs"
DEFAULT_INPUT_CANDIDATES = (
    Path("/tmp/lfg_waymo_front"),
    REPO_ROOT / "examples" / "frames",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class Job:
    id: str
    input_path: str
    checkpoint: str
    output_dir: str
    command: list[str]
    log_path: str
    status: str = "queued"
    created_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    returncode: int | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _default_input() -> str:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return ""


def _default_checkpoint() -> Path:
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_CHECKPOINT_CANDIDATES[-1]


def _safe_name(value: str, fallback: str = "file") -> str:
    name = value.replace("\\", "/").split("/")[-1].strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")
    return name or fallback


def _resolve_under(root: Path, *parts: str) -> Path:
    root = root.resolve()
    path = root.joinpath(*parts).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes output directory.")
    return path


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    *,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object.")
    return value


def _int_option(data: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = data.get(key, default)
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc


def _bool_option(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _make_command(data: dict[str, Any], job_id: str) -> Job:
    input_path = str(data.get("input_path") or "").strip()
    checkpoint = str(data.get("checkpoint") or _default_checkpoint()).strip()
    device = str(data.get("device") or "cuda:0").strip()
    resize_mode = str(data.get("resize_mode") or "crop").strip()
    target_size = _int_option(data, "target_size", 518)
    max_frames = _int_option(data, "max_frames", 3)
    window_stride = _int_option(data, "window_stride", 3)
    frame_stride = _int_option(data, "frame_stride", 1)
    start_frame = _int_option(data, "start_frame", 0)
    keep_ratio = _bool_option(data, "keep_ratio", False)

    if not input_path:
        raise ValueError("Input path is required.")
    if target_size is None or target_size <= 0:
        raise ValueError("target_size must be positive.")

    output_dir = RUN_ROOT / job_id
    log_path = output_dir / "job.log"
    command = [
        sys.executable,
        str(REPO_ROOT / "infer.py"),
        input_path,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--target-size",
        str(target_size),
        "--resize-mode",
        resize_mode,
    ]
    if frame_stride is not None:
        command.extend(["--frame-stride", str(frame_stride)])
    if start_frame is not None:
        command.extend(["--start-frame", str(start_frame)])
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    if window_stride is not None:
        command.extend(["--window-stride", str(window_stride)])
    if keep_ratio:
        command.append("--keep-ratio")

    return Job(
        id=job_id,
        input_path=input_path,
        checkpoint=checkpoint,
        output_dir=str(output_dir),
        command=command,
        log_path=str(log_path),
        created_at=_now(),
    )


def _set_job(job: Job) -> None:
    with JOBS_LOCK:
        JOBS[job.id] = job


def _get_job(job_id: str) -> Job | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _all_jobs() -> list[Job]:
    with JOBS_LOCK:
        return sorted(JOBS.values(), key=lambda job: job.created_at, reverse=True)


def _run_job(job_id: str) -> None:
    job = _get_job(job_id)
    if job is None:
        return

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"
    job.started_at = _now()
    _set_job(job)

    try:
        with Path(job.log_path).open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(job.command) + "\n\n")
            log.flush()
            process = subprocess.run(
                job.command,
                cwd=str(REPO_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        job.returncode = process.returncode
        job.status = "done" if process.returncode == 0 else "failed"
        if process.returncode != 0:
            job.error = f"infer.py exited with code {process.returncode}"
    except Exception as exc:  # pragma: no cover - defensive server boundary.
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.ended_at = _now()
        _set_job(job)


def _start_job(data: dict[str, Any]) -> Job:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job = _make_command(data, job_id)
    _set_job(job)
    thread = threading.Thread(target=_run_job, args=(job.id,), daemon=True)
    thread.start()
    return job


def _read_log(job: Job, max_chars: int = 12000) -> str:
    path = Path(job.log_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _file_url(job: Job, path: Path) -> str:
    rel = path.resolve().relative_to(Path(job.output_dir).resolve())
    return "/files/" + job.id + "/" + "/".join(rel.parts)


def _slot_labels(job: Job) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for metadata_path in sorted(Path(job.output_dir).glob("window_*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        window_name = metadata_path.parent.name
        for slot in metadata.get("predicted_slots", []):
            slot_id = f"{int(slot.get('slot', 0)):03d}"
            kind = str(slot.get("kind", "prediction"))
            nominal = slot.get("nominal_source_index")
            labels[(window_name, slot_id)] = f"{window_name}/{slot_id} {kind} idx {nominal}"
    return labels


def _list_pngs(job: Job, subdir: str) -> list[dict[str, str]]:
    base = Path(job.output_dir)
    labels = _slot_labels(job)
    items = []
    for path in sorted(base.glob(f"window_*/{subdir}/*.png")):
        window_name = path.parents[1].name
        slot_id = path.stem
        items.append(
            {
                "path": str(path),
                "url": _file_url(job, path),
                "label": labels.get((window_name, slot_id), "/".join(path.parts[-3:])),
            }
        )
    return items


def _result_payload(job: Job) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    metadata_path = output_dir / "run_metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None

    npz_files = []
    for path in sorted(output_dir.glob("window_*/predictions.npz")):
        npz_files.append({"path": str(path), "url": _file_url(job, path)})

    return {
        "metadata": metadata,
        "depth": _list_pngs(job, "depth"),
        "confidence": _list_pngs(job, "confidence"),
        "segmentation": _list_pngs(job, "segmentation"),
        "npz": npz_files,
        "motion": _list_pngs(job, "motion"),
        "flow": _list_pngs(job, "flow"),
    }


def _job_payload(job: Job, *, include_log: bool = False, include_results: bool = False) -> dict[str, Any]:
    payload = asdict(job)
    if include_log:
        payload["log"] = _read_log(job)
    if include_results:
        payload["results"] = _result_payload(job)
    return payload


def _parse_multipart_upload(handler: BaseHTTPRequestHandler) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")
    raw = handler.rfile.read(length)
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
    )

    files: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append((_safe_name(filename), payload))
        elif name:
            fields[name] = payload.decode("utf-8", errors="replace")
    return files, fields


def _save_upload(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    files, fields = _parse_multipart_upload(handler)
    if not files:
        raise ValueError("No files were uploaded.")

    upload_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    upload_dir = UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index, (filename, payload) in enumerate(files):
        path = upload_dir / _safe_name(filename, fallback=f"file_{index:04d}")
        if path.exists():
            stem = path.stem or "file"
            suffix = path.suffix
            path = upload_dir / f"{stem}_{index:04d}{suffix}"
        path.write_bytes(payload)
        saved.append(str(path))

    input_path = str(upload_dir)
    if len(saved) == 1 and Path(saved[0]).suffix.lower() in VIDEO_EXTENSIONS:
        input_path = saved[0]
    elif len(saved) == 1 and Path(saved[0]).suffix.lower() in IMAGE_EXTENSIONS:
        input_path = str(upload_dir)

    return {
        "upload_id": upload_id,
        "input_path": input_path,
        "saved_files": saved,
        "fields": fields,
    }


def _send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _html() -> str:
    default_checkpoint = str(_default_checkpoint())
    default_input = _default_input()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LFG Local Inference</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde5;
      --text: #171b22;
      --muted: #596273;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --ok: #087443;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      font-size: 18px;
      font-weight: 650;
      margin: 0;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 18px 32px;
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      gap: 18px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .panel-head {{
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    h2 {{
      font-size: 14px;
      font-weight: 650;
      margin: 0;
      letter-spacing: 0;
    }}
    .panel-body {{ padding: 14px; }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid #c9d0db;
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    input[type="checkbox"] {{
      width: auto;
      margin: 0 8px 0 0;
      transform: translateY(1px);
    }}
    input[type="file"] {{
      padding: 8px;
      background: #fbfcfe;
    }}
    .field {{ margin-bottom: 12px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px 12px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      min-height: 38px;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button.secondary {{
      background: #fff;
      color: var(--text);
      border-color: #c9d0db;
    }}
    button.secondary:hover {{ background: #f3f5f8; }}
    button:disabled {{
      opacity: .55;
      cursor: not-allowed;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 0 10px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
      font-weight: 650;
    }}
    .status.done {{ color: var(--ok); }}
    .status.failed {{ color: var(--danger); }}
    .jobs {{
      display: grid;
      gap: 8px;
    }}
    .job {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      background: #fbfcfe;
    }}
    .job.active {{
      border-color: var(--accent);
      background: #ecfdf9;
    }}
    .job-title {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 650;
      margin-bottom: 4px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 12px 14px 0;
      flex-wrap: wrap;
    }}
    .tab {{
      background: #fff;
      border: 1px solid #c9d0db;
      color: var(--text);
      min-height: 34px;
      padding: 6px 10px;
    }}
    .tab.active {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .results {{
      padding: 14px;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(156px, 1fr));
      gap: 10px;
    }}
    .thumb {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .thumb img {{
      width: 100%;
      aspect-ratio: 14 / 9;
      object-fit: cover;
      display: block;
      background: #eef1f5;
    }}
    .thumb a {{
      display: block;
      color: var(--text);
      text-decoration: none;
      padding: 7px 8px;
      font-size: 12px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      overflow: auto;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 8px;
      max-height: 360px;
      font-size: 12px;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px;
      border: 1px dashed #c9d0db;
      border-radius: 8px;
      background: #fbfcfe;
    }}
    .path {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      header {{ padding: 0 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>LFG Local Inference</h1>
    <span id="serverStatus" class="status">ready</span>
  </header>
  <main>
    <div>
      <section>
        <div class="panel-head">
          <h2>Run</h2>
          <button id="runButton">Run</button>
        </div>
        <div class="panel-body">
          <form id="uploadForm" class="field">
            <label for="files">Upload</label>
            <input id="files" name="files" type="file" multiple>
          </form>
          <div class="field row">
            <button id="uploadButton" class="secondary" type="button">Upload Files</button>
            <button id="useUploadButton" class="secondary" type="button" disabled>Use Upload</button>
          </div>
          <div class="field">
            <label for="inputPath">Input Path</label>
            <input id="inputPath" value="{default_input}" placeholder="/path/to/video-or-frame-directory">
          </div>
          <div class="field">
            <label for="checkpoint">Checkpoint</label>
            <input id="checkpoint" value="{default_checkpoint}">
          </div>
          <div class="grid-2">
            <div class="field">
              <label for="device">Device</label>
              <select id="device">
                <option value="cuda:0">cuda:0</option>
                <option value="cuda">cuda</option>
                <option value="auto">auto</option>
                <option value="cpu">cpu</option>
              </select>
            </div>
            <div class="field">
              <label for="resizeMode">Resize</label>
              <select id="resizeMode">
                <option value="crop">crop</option>
                <option value="pad">pad</option>
              </select>
            </div>
          </div>
          <div class="grid-2">
            <div class="field">
              <label for="targetSize">Target Size</label>
              <input id="targetSize" type="number" value="518" step="14" min="14">
            </div>
            <div class="field">
              <label for="maxFrames">Max Frames</label>
              <input id="maxFrames" type="number" value="3" min="1">
            </div>
          </div>
          <div class="grid-2">
            <div class="field">
              <label for="windowStride">Window Stride</label>
              <input id="windowStride" type="number" value="3" min="1">
            </div>
            <div class="field">
              <label for="frameStride">Frame Stride</label>
              <input id="frameStride" type="number" value="1" min="1">
            </div>
          </div>
          <div class="field row">
            <input id="keepRatio" type="checkbox">
            <label for="keepRatio" style="margin: 0;">Keep Ratio</label>
          </div>
        </div>
      </section>

      <section style="margin-top: 18px;">
        <div class="panel-head">
          <h2>Jobs</h2>
          <button id="refreshButton" class="secondary">Refresh</button>
        </div>
        <div class="panel-body">
          <div id="jobs" class="jobs"></div>
        </div>
      </section>
    </div>

    <section>
      <div class="panel-head">
        <h2>Results</h2>
        <span id="activeJob" class="muted">no job selected</span>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="depth">Depth</button>
        <button class="tab" data-tab="segmentation">Segmentation</button>
        <button class="tab" data-tab="motion">Motion</button>
        <button class="tab" data-tab="flow">Flow</button>
        <button class="tab" data-tab="confidence">Confidence</button>
        <button class="tab" data-tab="log">Log</button>
        <button class="tab" data-tab="metadata">Metadata</button>
      </div>
      <div id="results" class="results">
        <div class="empty">No results yet.</div>
      </div>
    </section>
  </main>

  <script>
    const state = {{ jobs: [], selectedJobId: null, tab: "depth", uploadedInput: null }};

    const $ = (id) => document.getElementById(id);

    function setStatus(text, cls = "") {{
      const node = $("serverStatus");
      node.textContent = text;
      node.className = "status " + cls;
    }}

    function requestBody() {{
      return {{
        input_path: $("inputPath").value.trim(),
        checkpoint: $("checkpoint").value.trim(),
        device: $("device").value,
        target_size: Number($("targetSize").value),
        max_frames: Number($("maxFrames").value),
        window_stride: Number($("windowStride").value),
        frame_stride: Number($("frameStride").value),
        resize_mode: $("resizeMode").value,
        keep_ratio: $("keepRatio").checked
      }};
    }}

    async function api(path, options = {{}}) {{
      const response = await fetch(path, options);
      const text = await response.text();
      let payload = null;
      try {{ payload = text ? JSON.parse(text) : null; }} catch (_) {{}}
      if (!response.ok) {{
        const detail = payload && payload.error ? payload.error : text;
        throw new Error(detail || response.statusText);
      }}
      return payload;
    }}

    async function refreshJobs(selectId = null) {{
      const payload = await api("/api/jobs");
      state.jobs = payload.jobs || [];
      if (selectId) state.selectedJobId = selectId;
      if (!state.selectedJobId && state.jobs.length) state.selectedJobId = state.jobs[0].id;
      renderJobs();
      if (state.selectedJobId) await loadJob(state.selectedJobId);
    }}

    function renderJobs() {{
      const root = $("jobs");
      if (!state.jobs.length) {{
        root.innerHTML = '<div class="empty">No jobs.</div>';
        return;
      }}
      root.innerHTML = state.jobs.map((job) => `
        <div class="job ${{job.id === state.selectedJobId ? "active" : ""}}" data-id="${{job.id}}">
          <div class="job-title">
            <span>${{job.status}}</span>
            <span class="muted">${{job.id.slice(9)}}</span>
          </div>
          <div class="muted path">${{job.input_path}}</div>
        </div>
      `).join("");
      root.querySelectorAll(".job").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.selectedJobId = node.dataset.id;
          renderJobs();
          loadJob(node.dataset.id);
        }});
      }});
    }}

    async function loadJob(jobId) {{
      const payload = await api(`/api/jobs/${{jobId}}`);
      state.selectedJob = payload;
      $("activeJob").textContent = `${{payload.status}} - ${{payload.id}}`;
      renderResults();
    }}

    function renderResults() {{
      const root = $("results");
      const job = state.selectedJob;
      if (!job) {{
        root.innerHTML = '<div class="empty">No job selected.</div>';
        return;
      }}
      if (state.tab === "log") {{
        root.innerHTML = `<pre>${{escapeHtml(job.log || "")}}</pre>`;
        return;
      }}
      if (state.tab === "metadata") {{
        const meta = job.results && job.results.metadata ? job.results.metadata : {{}};
        root.innerHTML = `<pre>${{escapeHtml(JSON.stringify(meta, null, 2))}}</pre>`;
        return;
      }}
      const items = job.results && job.results[state.tab] ? job.results[state.tab] : [];
      if (!items.length) {{
        root.innerHTML = `<div class="empty">No ${{state.tab}} images yet.</div>`;
        return;
      }}
      root.innerHTML = `<div class="thumb-grid">${{items.map((item) => `
        <div class="thumb">
          <a href="${{item.url}}" target="_blank"><img src="${{item.url}}" loading="lazy" alt=""></a>
          <a href="${{item.url}}" target="_blank">${{item.label || item.path.split("/").slice(-3).join("/")}}</a>
        </div>
      `).join("")}}</div>`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    async function runInference() {{
      setStatus("starting");
      $("runButton").disabled = true;
      try {{
        const payload = await api("/api/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(requestBody())
        }});
        state.selectedJobId = payload.job.id;
        await refreshJobs(payload.job.id);
        setStatus("running");
      }} catch (error) {{
        setStatus(error.message, "failed");
      }} finally {{
        $("runButton").disabled = false;
      }}
    }}

    async function uploadFiles() {{
      const files = $("files").files;
      if (!files.length) {{
        setStatus("choose files", "failed");
        return;
      }}
      const form = new FormData();
      for (const file of files) form.append("files", file, file.webkitRelativePath || file.name);
      setStatus("uploading");
      $("uploadButton").disabled = true;
      try {{
        const payload = await api("/api/upload", {{ method: "POST", body: form }});
        state.uploadedInput = payload.input_path;
        $("useUploadButton").disabled = false;
        $("inputPath").value = payload.input_path;
        setStatus("uploaded", "done");
      }} catch (error) {{
        setStatus(error.message, "failed");
      }} finally {{
        $("uploadButton").disabled = false;
      }}
    }}

    document.querySelectorAll(".tab").forEach((button) => {{
      button.addEventListener("click", () => {{
        state.tab = button.dataset.tab;
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        renderResults();
      }});
    }});

    $("runButton").addEventListener("click", runInference);
    $("uploadButton").addEventListener("click", uploadFiles);
    $("useUploadButton").addEventListener("click", () => {{
      if (state.uploadedInput) $("inputPath").value = state.uploadedInput;
    }});
    $("refreshButton").addEventListener("click", () => refreshJobs());

    setInterval(async () => {{
      try {{
        await refreshJobs();
        const selected = state.jobs.find((job) => job.id === state.selectedJobId);
        if (selected) setStatus(selected.status, selected.status === "failed" ? "failed" : selected.status === "done" ? "done" : "");
      }} catch (_) {{}}
    }}, 3000);

    refreshJobs().catch((error) => setStatus(error.message, "failed"));
  </script>
</body>
</html>
"""


class LFGWebHandler(BaseHTTPRequestHandler):
    server_version = "LFGWeb/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._do_get()
        except ValueError as exc:
            _json_response(self, {"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - defensive server boundary.
            _json_response(self, {"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._do_post()
        except ValueError as exc:
            _json_response(self, {"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - defensive server boundary.
            _json_response(self, {"error": str(exc)}, status=500)

    def _do_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            _text_response(self, _html(), content_type="text/html; charset=utf-8")
            return
        if path == "/api/status":
            _json_response(
                self,
                {
                    "ok": True,
                    "repo_root": str(REPO_ROOT),
                    "web_root": str(WEB_ROOT),
                    "default_checkpoint": str(_default_checkpoint()),
                    "default_input": _default_input(),
                },
            )
            return
        if path == "/api/jobs":
            _json_response(self, {"jobs": [_job_payload(job) for job in _all_jobs()]})
            return
        if path.startswith("/api/jobs/"):
            job_id = unquote(path.removeprefix("/api/jobs/")).strip("/")
            job = _get_job(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            _json_response(self, _job_payload(job, include_log=True, include_results=True))
            return
        if path.startswith("/files/"):
            parts = path.removeprefix("/files/").split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            job_id, rel = parts
            job = _get_job(unquote(job_id))
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            file_path = _resolve_under(Path(job.output_dir), *unquote(rel).split("/"))
            _send_file(self, file_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _do_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            data = _read_json(self)
            job = _start_job(data)
            _json_response(self, {"job": _job_payload(job)}, status=202)
            return
        if path == "/api/upload":
            payload = _save_upload(self)
            _json_response(self, payload, status=201)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def run_server(host: str, port: int) -> None:
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((host, port), LFGWebHandler)
    print(f"LFG web UI listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local LFG inference web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for Tailscale/LAN access.")
    parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
