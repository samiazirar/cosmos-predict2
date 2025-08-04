"""
cosmos_video_service.py
-----------------------

A FastAPI service that mimics the OpenAI `/v1/chat/completions` endpoint but,
instead of returning text, responds with a URL pointing to a video generated
by Cosmos Predict2.

Input is **100 % OpenAI-compatible**: each message `content` element may be
either plain text or an **`image_url`**.  
If the URL points at an *image*, single-frame conditioning is used.  
If the URL points at a *video* **and** `num_conditional_frames` ≥ 5,
multi-frame conditioning is used.  

Rules
~~~~~
* `num_conditional_frames == 1` (default) → behave exactly as before
  (image or last frame of video).
* `num_conditional_frames` in **2 … 4** → 400 Bad Request.
* `num_conditional_frames ≥ 5` → requires a video input; the model is run
  with that many conditioning frames (defaults to **5** when omitted).

The client code for *image* workflows is completely unaffected.

--------------------------------------------------------------------
Quick start
--------------------------------------------------------------------
pip install "fastapi[all]" pillow python-multipart requests opencv-python-headless
uvicorn cosmos_video_service:app --port 8001
--------------------------------------------------------------------
"""

import os
import io
import time
import uuid
import base64
from typing import List, Dict, Any, Optional

import cv2                                          # ← NEW (used to grab last frame)
import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse
from PIL import Image

# Cosmos generator
from cosmos_video_generator import generate_cosmos_video

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OUT_DIR = os.getenv("OUT_DIR", "/tmp/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COSMOS_MODEL_SIZE = os.getenv("COSMOS_MODEL_SIZE", "14B")
COSMOS_NUM_GPUS = int(os.getenv("COSMOS_NUM_GPUS", "4"))
COSMOS_DISABLE_GUARDRAIL = os.getenv("COSMOS_DISABLE_GUARDRAIL", "true").lower() == "true"
COSMOS_DISABLE_PROMPT_REFINER = os.getenv("COSMOS_DISABLE_PROMPT_REFINER", "true").lower() == "true"

DEFAULT_NUM_CONDITIONAL_FRAMES = int(os.getenv("COSMOS_NUM_CONDITIONAL_FRAMES", "5"))
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# --------------------------------------------------------------------------- #
# Helpers (images & videos)
# --------------------------------------------------------------------------- #


def _decode_data_uri(uri: str) -> bytes:
    header, _, b64data = uri.partition(",")
    if not header.startswith("data:") or not b64data:
        raise ValueError("Invalid data URI")
    return base64.b64decode(b64data)


def _is_video_url(url: str) -> bool:
    return url.lower().endswith(_VIDEO_EXTENSIONS)


def _fetch_image(url: str) -> Image.Image:
    """Download / decode an image URL and return RGB PIL.Image."""
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


def _download_video(url: str) -> str:
    """Download remote video to a temp file and return its path."""
    try:
        # Handle local temp video URLs
        if url.startswith("/temp_video/"):
            file_id = url.split("/temp_video/")[1]
            temp_dir = os.path.join(OUT_DIR, "temp_uploads")
            local_path = os.path.join(temp_dir, file_id)
            if os.path.exists(local_path):
                return local_path
            else:
                raise HTTPException(404, f"Temporary video file not found: {file_id}")
        
        # Handle remote URLs
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        temp_dir = os.path.join(OUT_DIR, "temp_inputs")
        os.makedirs(temp_dir, exist_ok=True)
        tmp_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.mp4")
        with open(tmp_path, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=8192):
                fp.write(chunk)
        return tmp_path
    except Exception as e:
        raise HTTPException(422, f"Cannot load video: {e}") from e


def _extract_last_frame(video_path: str) -> Image.Image:
    """Grab the last frame of a video as a PIL.Image (RGB)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video for frame extraction")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        raise HTTPException(500, "Video appears to have zero frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise HTTPException(500, "Failed to read last frame from video")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _save_temp_image(image: Image.Image) -> str:
    temp_dir = os.path.join(OUT_DIR, "temp_inputs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.jpg")
    image.save(temp_path, "JPEG", quality=95)
    return temp_path


# --------------------------------------------------------------------------- #
# Generation wrappers
# --------------------------------------------------------------------------- #


def _gen_single_frame(
    prompt: str,
    image: Optional[Image.Image],
    **kwargs,
) -> str:
    """
    Standard single-frame conditioning helper.

    If `image` is None a default asset will be used.
    """
    if image is not None:
        input_path = _save_temp_image(image)
    else:
        input_path = "assets/video2world/input0.jpg"
        if not os.path.exists(input_path):
            raise HTTPException(500, "No input image provided and default input not found")

    try:
        output_path = os.path.join(OUT_DIR, f"cosmos_{uuid.uuid4().hex}.mp4")
        result = generate_cosmos_video(
            prompt=prompt,
            input_path=input_path,
            save_path=output_path,
            model_size=COSMOS_MODEL_SIZE,
            num_gpus=COSMOS_NUM_GPUS,
            disable_guardrail=COSMOS_DISABLE_GUARDRAIL,
            disable_prompt_refiner=COSMOS_DISABLE_PROMPT_REFINER,
            capture_output=False,  # Don't capture output to show progress bars
            **kwargs,
        )
        if result["success"]:
            return result["output_path"]
        raise HTTPException(500, f"Cosmos generation failed: {result['error']}")
    finally:
        if image is not None and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


def _gen_multi_frame(
    prompt: str,
    video_path: str,
    num_conditional_frames: int,
    **kwargs,
) -> str:
    """Video2World multi-frame conditioning helper."""
    if not os.path.exists(video_path):
        raise HTTPException(400, "Video path does not exist")

    try:
        output_path = os.path.join(OUT_DIR, f"cosmos_{uuid.uuid4().hex}.mp4")
        print(f"Num cond. frames: {num_conditional_frames}")
        result = generate_cosmos_video(
            prompt=prompt,
            input_path=video_path,
            save_path=output_path,
            model_size=COSMOS_MODEL_SIZE,
            num_gpus=COSMOS_NUM_GPUS,
            disable_guardrail=COSMOS_DISABLE_GUARDRAIL,
            disable_prompt_refiner=COSMOS_DISABLE_PROMPT_REFINER,
            capture_output=False,  # Don't capture output to show progress bars
            num_conditional_frames=num_conditional_frames,
            **kwargs,
        )
        if result["success"]:
            return result["output_path"]
        raise HTTPException(500, f"Cosmos generation failed: {result['error']}")
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# OpenAI-style request / response models
# --------------------------------------------------------------------------- #


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

    class Config:
        extra = "allow"


class ChatMessage(BaseModel):
    role: str
    content: List[MessageContent]

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="cosmos-video-001")
    messages: List[ChatMessage]
    num_conditional_frames: Optional[int] = Field(default=None, description="Number of conditional frames for video generation")

    # **Unknown top-level keys are accepted**
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
    title="Cosmos Video Generation Service",
    version="2.0.0",
    description="Generates videos using Cosmos Predict2 (single- or multi-frame) "
                "through an OpenAI-compatible endpoint.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "model": COSMOS_MODEL_SIZE,
        "gpus": COSMOS_NUM_GPUS,
        "default_num_conditional_frames": DEFAULT_NUM_CONDITIONAL_FRAMES,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cosmos_config": {
            "model_size": COSMOS_MODEL_SIZE,
            "num_gpus": COSMOS_NUM_GPUS,
            "disable_guardrail": COSMOS_DISABLE_GUARDRAIL,
            "disable_prompt_refiner": COSMOS_DISABLE_PROMPT_REFINER,
            "default_conditional_frames": DEFAULT_NUM_CONDITIONAL_FRAMES,
        },
        "output_dir": OUT_DIR,
    }


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest, http_request: Request):
    """
    OpenAI-compatible chat-completion that returns a *video* URL.
    Supports single-frame (images) **and** multi-frame (videos).
    """

    # 1 — the last user message
    try:
        last = next(m for m in reversed(req.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(400, "At least one user message is required")

    # 2 — parse parts
    prompt_parts: list[str] = []
    image: Optional[Image.Image] = None
    video_path: Optional[str] = None

    for part in last.content:
        if part.type == "text" and part.text:
            prompt_parts.append(part.text)
        elif part.type == "image_url" and part.image_url:
            url = part.image_url.get("url", "")
            if not url:
                continue
            if _is_video_url(url) and video_path is None:
                video_path = _download_video(url)
            elif image is None:
                image = _fetch_image(url)   # image or data-URI

    prompt = " ".join(prompt_parts).strip() or "(empty prompt)"

    # 3 — determine num_conditional_frames
    if req.num_conditional_frames is None:
        num_conditional_frames = DEFAULT_NUM_CONDITIONAL_FRAMES if video_path else 1
    else:
        try:
            num_conditional_frames = int(req.num_conditional_frames)
        except ValueError:
            raise HTTPException(400, "num_conditional_frames must be an integer")

    # 4 — route to generator according to the rules
    if num_conditional_frames == 1:
        # image path given → normal flow
        if video_path and image is None:          # video provided but 1-frame requested
            image = _extract_last_frame(video_path)
        video_path_final = _gen_single_frame(prompt, image)
    elif 2 <= num_conditional_frames <= 4:
        raise HTTPException(
            400,
            "num_conditional_frames must be 1 or ≥ 5 "
            "(2–4-frame conditioning is unsupported)",
        )
    else:  # ≥ 5
        if not video_path:
            raise HTTPException(
                400,
                "Multi-frame conditioning (num_conditional_frames ≥ 5) "
                "requires a video input",
            )
        video_path_final = _gen_multi_frame(
            prompt,
            video_path,
            num_conditional_frames=num_conditional_frames,
        )

    # 5 — build OpenAI-style response
    filename = os.path.basename(video_path_final)
    video_url = str(http_request.url_for("download", file_id=filename))

    choice = Choice(
        index=0,
        message={
            "role": "assistant",
            "content": video_url,
            "metadata": {
                "generated_with": "cosmos-predict2",
                "model_size": COSMOS_MODEL_SIZE,
                "prompt": prompt,
                "num_conditional_frames": num_conditional_frames,
            },
        },
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
    )


# --------------------------------------------------------------------------- #
# File upload endpoint for videos
# --------------------------------------------------------------------------- #

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and return a temporary URL that can be used 
    with the video generation endpoints.
    """
    if not file.filename.lower().endswith(_VIDEO_EXTENSIONS):
        raise HTTPException(400, f"File must be a video with extension: {_VIDEO_EXTENSIONS}")
    
    # Save uploaded file to temp directory
    temp_dir = os.path.join(OUT_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_id = f"upload_{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(temp_dir, file_id)
    
    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Return a URL that can be used in video generation requests
        video_url = f"/temp_video/{file_id}"
        return {
            "success": True,
            "video_url": video_url,
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content)
        }
    except Exception as e:
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except:
                pass
        raise HTTPException(500, f"Failed to upload file: {e}")


@app.get("/temp_video/{file_id}")
def get_temp_video(file_id: str):
    """Serve temporarily uploaded video files."""
    temp_dir = os.path.join(OUT_DIR, "temp_uploads")
    path = os.path.join(temp_dir, file_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Temporary video file not found")
    return FileResponse(path, media_type="video/mp4")


# --------------------------------------------------------------------------- #
# /download
# --------------------------------------------------------------------------- #

@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(OUT_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")


# --------------------------------------------------------------------------- #
# Legacy direct endpoint (non-OpenAI) – now with conditional-frame support
# --------------------------------------------------------------------------- #

@app.post("/generate")
def generate_direct(
    prompt: str,
    image_url: Optional[str] = None,
    model_size: Optional[str] = None,
    num_gpus: Optional[int] = None,
    num_conditional_frames: int = 1,
):
    """
    Convenience endpoint that mirrors the rules used by `/v1/chat/completions`.
    """

    _model_size = model_size or COSMOS_MODEL_SIZE
    _num_gpus = num_gpus or COSMOS_NUM_GPUS

    image: Optional[Image.Image] = None
    video_path: Optional[str] = None

    if image_url:
        if _is_video_url(image_url):
            video_path = _download_video(image_url)
        else:
            image = _fetch_image(image_url)

    # routing identical to the chat endpoint
    if num_conditional_frames == 1:
        if video_path and image is None:
            image = _extract_last_frame(video_path)
        video_path_final = _gen_single_frame(
            prompt,
            image,
            model_size=_model_size,
            num_gpus=_num_gpus,
            capture_output=False,  # Don't capture output to show progress bars
        )
    elif 2 <= num_conditional_frames <= 4:
        raise HTTPException(
            400,
            "num_conditional_frames must be 1 or ≥ 5 "
            "(2–4-frame conditioning is unsupported)",
        )
    else:
        if not video_path:
            raise HTTPException(
                400,
                "Multi-frame conditioning (num_conditional_frames ≥ 5) "
                "requires a video input",
            )
        video_path_final = _gen_multi_frame(
            prompt,
            video_path,
            num_conditional_frames=num_conditional_frames,
            model_size=_model_size,
            num_gpus=_num_gpus,
            capture_output=False,  # Don't capture output to show progress bars
        )

    return {
        "success": True,
        "video_path": video_path_final,
        "download_url": f"/download/{os.path.basename(video_path_final)}",
        "prompt": prompt,
        "model_size": _model_size,
        "num_conditional_frames": num_conditional_frames,
    }


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
