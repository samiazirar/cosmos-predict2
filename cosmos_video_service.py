"""cosmos_video_service.py
A FastAPI service that mimics the OpenAI `/v1/chat/completions` endpoint but,
instead of returning text, responds with a URL pointing to a video generated
using Cosmos Predict2.

Inputs are **100% OpenAI-compatible**: every message `content` can include any mix of:
    {"type": "text",       "text": "..."}               # normal text
    {"type": "image_url",  "image_url": {"url": ...}}   # image (data-URI or HTTP)

That means you can drop this straight into existing multimodal clients and
simply change the base-URL.

-------------------------------------------------------------------
Quick start
-------------------------------------------------------------------
pip install "fastapi[all]" pillow python-multipart requests
uvicorn cosmos_video_service:app --port 8001
-------------------------------------------------------------------
"""

import os
import io
import time
import uuid
import base64
import tempfile
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse
from PIL import Image

# Import our Cosmos video generator
from cosmos_video_generator import generate_cosmos_video

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OUT_DIR = os.getenv("OUT_DIR", "/tmp/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Cosmos configuration
COSMOS_MODEL_SIZE = os.getenv("COSMOS_MODEL_SIZE", "14B")
COSMOS_NUM_GPUS = int(os.getenv("COSMOS_NUM_GPUS", "4"))
COSMOS_DISABLE_GUARDRAIL = os.getenv("COSMOS_DISABLE_GUARDRAIL", "true").lower() == "true"
COSMOS_DISABLE_PROMPT_REFINER = os.getenv("COSMOS_DISABLE_PROMPT_REFINER", "true").lower() == "true"

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


def _save_temp_image(image: Image.Image) -> str:
    """Save PIL Image to a temporary file and return the path."""
    temp_dir = os.path.join(OUT_DIR, "temp_inputs")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.jpg")
    image.save(temp_path, "JPEG", quality=95)
    return temp_path


def generate_cosmos_video_from_prompt(
    prompt: str,
    image: Optional[Image.Image] = None,
    **kwargs
) -> str:
    """Generate a video using Cosmos Predict2 and return the absolute path."""
    
    # Handle input conditioning
    if image is not None:
        # Save the image to a temporary file
        input_path = _save_temp_image(image)
    else:
        # Use a default input if no image is provided
        # You might want to have a default image in your assets
        input_path = "assets/video2world/input0.jpg"
        if not os.path.exists(input_path):
            raise HTTPException(500, "No input image provided and default input not found")
    
    try:
        # Generate output path
        output_filename = f"cosmos_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUT_DIR, output_filename)
        
        # Call the Cosmos generator
        result = generate_cosmos_video(
            prompt=prompt,
            input_path=input_path,
            save_path=output_path,
            model_size=COSMOS_MODEL_SIZE,
            num_gpus=COSMOS_NUM_GPUS,
            disable_guardrail=COSMOS_DISABLE_GUARDRAIL,
            disable_prompt_refiner=COSMOS_DISABLE_PROMPT_REFINER,
            **kwargs
        )
        
        if result["success"]:
            return result["output_path"]
        else:
            raise HTTPException(500, f"Cosmos generation failed: {result['error']}")
            
    except Exception as e:
        raise HTTPException(500, f"Video generation error: {str(e)}") from e
    finally:
        # Clean up temporary input file if it was created
        if image is not None and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass  # Ignore cleanup errors


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
    model: str = Field(default="cosmos-video-001")
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
    title="Cosmos Video Generation Service",
    version="1.0.0",
    description="Generates videos using Cosmos Predict2 through an OpenAI-compatible endpoint.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {"status": "ok", "model": COSMOS_MODEL_SIZE, "gpus": COSMOS_NUM_GPUS}


@app.get("/health")
def health():
    """Health check endpoint with configuration info."""
    return {
        "status": "healthy",
        "cosmos_config": {
            "model_size": COSMOS_MODEL_SIZE,
            "num_gpus": COSMOS_NUM_GPUS,
            "disable_guardrail": COSMOS_DISABLE_GUARDRAIL,
            "disable_prompt_refiner": COSMOS_DISABLE_PROMPT_REFINER,
        },
        "output_dir": OUT_DIR
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest, http_request: Request):
    """Return a *video* (URL) instead of text, generated using Cosmos Predict2."""

    # 1. Find the last user message
    try:
        last = next(m for m in reversed(req.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(400, "At least one user message is required")

    # 2. Parse message parts
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

    # 3. Generate the video using Cosmos
    try:
        video_path = generate_cosmos_video_from_prompt(prompt, image)
        video_filename = os.path.basename(video_path)
        video_url = str(http_request.url_for("download", file_id=video_filename))
        
        # 4. Build OpenAI-style response
        choice = Choice(
            index=0,
            message={
                "role": "assistant", 
                "content": video_url,
                "metadata": {
                    "generated_with": "cosmos-predict2",
                    "model_size": COSMOS_MODEL_SIZE,
                    "prompt": prompt
                }
            },
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[choice],
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {str(e)}") from e


@app.get("/download/{file_id}")
def download(file_id: str):
    """Download generated video files."""
    path = os.path.join(OUT_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/generate")
def generate_direct(
    prompt: str,
    image_url: Optional[str] = None,
    model_size: Optional[str] = None,
    num_gpus: Optional[int] = None
):
    """Direct video generation endpoint (non-OpenAI compatible)."""
    
    # Handle optional parameters
    _model_size = model_size or COSMOS_MODEL_SIZE
    _num_gpus = num_gpus or COSMOS_NUM_GPUS
    
    # Handle image input
    image = None
    if image_url:
        image = _fetch_image(image_url)
    
    # Generate video
    try:
        video_path = generate_cosmos_video_from_prompt(
            prompt, 
            image,
            model_size=_model_size,
            num_gpus=_num_gpus
        )
        
        return {
            "success": True,
            "video_path": video_path,
            "download_url": f"/download/{os.path.basename(video_path)}",
            "prompt": prompt,
            "model_size": _model_size
        }
        
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
