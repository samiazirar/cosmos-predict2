"""dummy_video_service.py
A self-contained FastAPI service that mimics the OpenAI
`/v1/chat/completions` endpoint but, instead of returning text,
responds with a URL pointing to a short MP4 it generates on-the-fly.

Inputs are **100 % OpenAI-compatible**: every message `content`
can include any mix of:

    {"type": "text",       "text": "..."}               # normal text
    {"type": "image_url",  "image_url": {"url": ...}}   # image (data-URI or HTTP)

That means you can drop this straight into existing multimodal
clients and simply change the base-URL.

-------------------------------------------------------------------
Quick start
-------------------------------------------------------------------
pip install "fastapi[all]" pillow imageio python-multipart requests
uvicorn dummy_video_service:app --port 8001
-------------------------------------------------------------------
"""

import os
import io
import time
import uuid
import base64
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OUT_DIR = os.getenv("OUT_DIR", "/tmp/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _decode_data_uri(uri: str) -> bytes:
    """Return raw bytes from a `data:` URI."""
    header, _, b64data = uri.partition(",")
    if not header.startswith("data:") or not b64data:
        raise ValueError("Invalid data URI")
    return base64.b64decode(b64data)


def _fetch_image(url: str) -> Image.Image:
    """Download **or** decode an image and return a RGB PIL.Image."""
    try:
        if url.startswith("data:"):
            img_bytes = _decode_data_uri(url)
        else:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Cannot load image: {e}") from e


def generate_dummy_video(
    prompt: str,
    image: Optional[Image.Image] = None,
    seconds: int = 3,
    fps: int = 10,
) -> str:
    """Generate a tiny looping MP4 and return its absolute path."""
    width, height = (image.size if image else (512, 512))
    n_frames = seconds * fps
    font = ImageFont.load_default()

    frames = []
    for i in range(n_frames):
        frame = (image.copy() if image is not None
                 else Image.new("RGB", (width, height), "black"))

        draw = ImageDraw.Draw(frame)
        # scroll the prompt vertically
        y = (height + 20) - (i * 2) % (height + 20)
        draw.text((10, y), prompt, fill="white", font=font)
        frames.append(np.asarray(frame))

    outfile = os.path.join(OUT_DIR, f"{uuid.uuid4().hex}.mp4")
    iio.imwrite(outfile, np.stack(frames), fps=fps, codec="libx264")
    return outfile

# --------------------------------------------------------------------------- #
# OpenAI-style request / response models
# --------------------------------------------------------------------------- #


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

    class Config:
        extra = "allow"     # silently ignore unknown keys


class ChatMessage(BaseModel):
    role: str
    content: List[MessageContent]

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="dummy-video-001")
    messages: List[ChatMessage]

    class Config:
        extra = "allow"


class Choice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Dummy Video Generation Service",
    version="0.2.0",
    description="Creates tiny MP4s through an OpenAI-compatible endpoint.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest, http_request: Request):
    """Return a *video* (URL) instead of text."""

    # 1. Last user message
    try:
        last = next(m for m in reversed(req.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(400, "At least one user message is required")

    # 2. Parse parts
    prompt_parts: list[str] = []
    image: Optional[Image.Image] = None
    for part in last.content:
        if part.type == "text" and part.text:
            prompt_parts.append(part.text)
        elif part.type == "image_url" and part.image_url and image is None:
            url = part.image_url.get("url", "")
            if url:
                image = _fetch_image(url)

    prompt = " ".join(prompt_parts).strip() or "(empty prompt)"

    # 3. Generate the video
    video_path = generate_dummy_video(prompt, image)
    video_url = str(http_request.url_for(
        "download", file_id=os.path.basename(video_path)))

    # 4. Build OpenAI-style response
    choice = Choice(
        index=0,
        message={"role": "assistant", "content": video_url},
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
    )


@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(OUT_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")
# --------------------------------------------------------------------------- #