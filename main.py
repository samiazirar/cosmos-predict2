"""
Cosmos‑Predict2 (or any HF video‑gen pipeline) → REST service.

Change behaviour with env‑vars:
    MODEL_ID   = nvidia/Cosmos-Predict2-14B-Video2World  (default)
    TP_SIZE    = 4   # shard across 4 GPUs with DeepSpeed
    MAX_BATCH  = 2   # batch size per request
    OUT_DIR    = /tmp/outputs
"""
import os, io, uuid, shutil, torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from diffusers import Cosmos2VideoToWorldPipeline           # HF snippet :contentReference[oaicite:1]{index=1}
import deepspeed                                            # auto‑TP :contentReference[oaicite:2]{index=2}
from tempfile import NamedTemporaryFile

MODEL_ID = os.getenv("MODEL_ID", "nvidia/Cosmos-Predict2-14B-Video2World")
TP_SIZE  = int(os.getenv("TP_SIZE", "1"))
OUT_DIR  = os.getenv("OUT_DIR", "/tmp/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load once (tensor‑parallel if TP_SIZE>1) ----------
base_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
if TP_SIZE > 1:
    pipe = deepspeed.init_inference(                        # shards weights :contentReference[oaicite:3]{index=3}
        base_pipe,
        mp_size=TP_SIZE,
        dtype=torch.float16,
        replace_method="auto",
    )
else:
    pipe = base_pipe.to("cuda")

# pipe.enable_model_cpu_offload()  # keeps VRAM tidy on ≤2 GPUs

# ---------- FastAPI ----------
app = FastAPI(
    title="Cosmos‑Predict2 Video2World",
    version="0.1.0",
    summary=f"Serving {MODEL_ID}",
)

class GenResponse(BaseModel):
    video_path: str

@app.post("/generate", response_model=GenResponse)
async def generate(prompt: str = Form(...),
                   image: UploadFile = File(...)):
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(422, "Expected image/* upload")

    raw = await image.read()
    first_frame = Image.open(io.BytesIO(raw)).convert("RGB")

    seed = torch.seed()
    video = pipe(prompt=prompt,
                 image=first_frame,
                 generator=torch.Generator(device="cuda").manual_seed(seed),
                 num_inference_steps=20)["video"]      # 5‑sec clip default

    tmp = NamedTemporaryFile(suffix=".mp4", dir=OUT_DIR, delete=False)
    video.write_videofile(tmp.name, codec="libx264", fps=16)
    return {"video_path": tmp.name}

@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(OUT_DIR, file_id + ".mp4")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")

# health‑check
@app.get("/ping")
def ping():
    return {"status": "ok", "model": MODEL_ID}
