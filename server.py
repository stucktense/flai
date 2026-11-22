"""
Wan 2.2 video server: web page -> FastAPI -> ComfyUI.

Needs: pip install fastapi uvicorn httpx python-multipart ; ffmpeg installed ; ComfyUI running with Wan 2.2.
Files next to this one (export each from ComfyUI with "Save (API Format)"):
  workflow_t2v.json  text-to-video workflow
  workflow_i2v.json  image-to-video workflow (optional; used to continue a scene from the last frame)
Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio, copy, json, os, random, subprocess, time, uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

COMFY = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
PUBLIC = os.getenv("PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/")
ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://yourusername.github.io")  # your site, no trailing slash
PROMPT_NODE = os.getenv("PROMPT_NODE", "6")      # id of the text prompt node in the workflows
PROMPT_FIELD = os.getenv("PROMPT_FIELD", "text")
IMAGE_NODE = os.getenv("IMAGE_NODE", "52")       # id of the LoadImage node in the image-to-video workflow
RATE_PER_HOUR = int(os.getenv("RATE_PER_HOUR", "10"))
TIMEOUT_S = 25 * 60

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
T2V = json.load(open("workflow_t2v.json"))
I2V = json.load(open("workflow_i2v.json")) if Path("workflow_i2v.json").exists() else None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.mount("/videos", StaticFiles(directory=OUT), name="videos")

jobs, hits, LOCK = {}, {}, asyncio.Lock()   # one video at a time; others wait in line


class Req(BaseModel):
    prompt: str
    previous_video_url: str | None = None


def randomize_seeds(wf):
    for node in wf.values():
        for k in ("seed", "noise_seed"):
            if k in node.get("inputs", {}) and not isinstance(node["inputs"][k], list):
                node["inputs"][k] = random.randint(0, 2**53)


def last_frame(video_url):
    # Only our own files: take the file name, never fetch arbitrary URLs.
    if not video_url.startswith(PUBLIC + "/videos/"):
        raise ValueError("previous video is not from this server")
    src = OUT / Path(video_url).name
    dst = OUT / (src.stem + "_last.png")
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(src), "-frames:v", "1", str(dst)],
                   check=True, capture_output=True)
    return dst


async def comfy_run(client, wf):
    r = await client.post(f"{COMFY}/prompt", json={"prompt": wf})
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    start = time.time()
    while time.time() - start < TIMEOUT_S:
        await asyncio.sleep(3)
        hist = (await client.get(f"{COMFY}/history/{pid}")).json()
        if pid in hist and hist[pid].get("outputs"):
            for node in hist[pid]["outputs"].values():
                for key in ("videos", "gifs", "images"):
                    for f in node.get(key, []):
                        if f["filename"].lower().endswith((".mp4", ".webm", ".webp", ".gif")):
                            return f
            raise RuntimeError("ComfyUI finished but returned no video file")
    raise RuntimeError("Timed out waiting for ComfyUI")


async def run_job(jid, prompt, prev):
    job = jobs[jid]
    try:
        async with LOCK:
            job["status"] = "running"
            async with httpx.AsyncClient(timeout=60) as client:
                wf = copy.deepcopy(T2V)
                if prev and I2V:
                    frame = await asyncio.to_thread(last_frame, prev)
                    with open(frame, "rb") as fh:
                        up = await client.post(f"{COMFY}/upload/image", files={"image": (frame.name, fh, "image/png")})
                    up.raise_for_status()
                    wf = copy.deepcopy(I2V)
                    wf[IMAGE_NODE]["inputs"]["image"] = up.json()["name"]
                wf[PROMPT_NODE]["inputs"][PROMPT_FIELD] = prompt
                randomize_seeds(wf)
                f = await comfy_run(client, wf)
                data = await client.get(f"{COMFY}/view", params={"filename": f["filename"],
                                        "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                data.raise_for_status()
                name = f"{jid}{Path(f['filename']).suffix}"
                (OUT / name).write_bytes(data.content)
        job.update(status="done", video_url=f"{PUBLIC}/videos/{name}")
    except Exception as e:
        job.update(status="error", error=str(e)[:200])


@app.post("/generate")
async def generate(req: Req, request: Request):
    ip = request.headers.get("cf-connecting-ip") or request.client.host
    now = time.time()
    hits[ip] = [t for t in hits.get(ip, []) if now - t < 3600]
    if len(hits[ip]) >= RATE_PER_HOUR:
        raise HTTPException(429, "Too many requests. Try again later.")
    prompt = req.prompt.strip()
    if not prompt or len(prompt) > 1000:
        raise HTTPException(400, "Prompt must be 1 to 1000 characters.")
    hits[ip].append(now)
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = {"status": "queued"}
    asyncio.create_task(run_job(jid, prompt, req.previous_video_url))
    return {"job_id": jid}


@app.get("/status/{jid}")
async def status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[jid]
