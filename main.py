"""
Cosmos‑Predict2 (Video2World) REST service.

This service exposes a simple FastAPI endpoint for generating videos from
text descriptions and an input image or video.  It uses NVIDIA's
Cosmos‑Predict2 Video2World pipeline under the hood, which supports
multi‑GPU context parallelism for faster inference.  You can control
the behaviour of the service through a few environment variables:

* ``MODEL_SIZE`` – either ``"2B"`` or ``"14B"`` to select the model size
  (defaults to ``"2B"``).  Larger models generally produce higher quality
  but require more GPU memory.
* ``NUM_GPUS`` – the number of GPUs to use for context‑parallel inference
  (defaults to ``1``).  For multi‑GPU inference you should launch the
  container with ``torchrun --nproc_per_node=${NUM_GPUS}`` and set
  ``NUM_GPUS`` accordingly.  See the official docs for details【505310102995698†L595-L633】.
* ``OUT_DIR`` – directory where generated videos will be written
  (defaults to ``/tmp/outputs``).

At start‑up the script initialises the Video2World pipeline once and
keeps it in memory.  Incoming requests run generation in a separate
thread to avoid blocking the FastAPI event loop.  The service returns
the file name of the generated video, which can subsequently be
downloaded via the ``/download`` endpoint.
"""

import os
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import torch

# Import Cosmos‑Predict2 components.  These modules live inside the
# ``cosmos_predict2`` repository and expose configuration objects for
# each model size.  The pipeline class encapsulates the full
# video‑generation stack including the guardrail and prompt refiner.
try:
    from cosmos_predict2.configs.base.config_video2world import (
        PREDICT2_VIDEO2WORLD_PIPELINE_2B,
        PREDICT2_VIDEO2WORLD_PIPELINE_14B,
    )
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from imaginaire.utils.io import save_image_or_video
    from megatron.core import parallel_state
except ImportError as e:
    # Provide a clear error message if the user hasn't installed the
    # necessary dependencies.  The service will not start until these
    # packages are available.
    raise ImportError(
        "Missing cosmos_predict2 dependencies. Ensure that the cosmos-predict2 "
        "repository is installed in your environment."
    ) from e


# ---------------------------------------------------------------------------
# Environment configuration
#
# Read environment variables early so that the container can be configured
# without editing this file.  Defaults are chosen to run the 2B model on a
# single GPU and write outputs to /tmp/outputs.
MODEL_SIZE = os.getenv("MODEL_SIZE", "2B")
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
OUT_DIR = os.getenv("OUT_DIR", "/tmp/outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline setup
#
# Select the appropriate configuration based on the requested model size.
if MODEL_SIZE.upper() == "14B":
    _config = PREDICT2_VIDEO2WORLD_PIPELINE_14B
else:
    # Fallback to 2B when an unknown size is specified
    _config = PREDICT2_VIDEO2WORLD_PIPELINE_2B

# Initialise context parallelism if multiple GPUs are requested.  The
# Video2World pipeline uses Megatron's ``parallel_state`` to split the
# video across GPUs.  The documentation states that ``--nproc_per_node``
# (handled by torchrun) and ``--num_gpus`` (this variable) should be
# equal【505310102995698†L595-L633】.  When running inside a single process (e.g. testing or
# single‑GPU Docker), ``initialize_model_parallel`` is safe to call
# with ``world_size=1``.
if NUM_GPUS > 1:
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(NUM_GPUS)

# Create the Video2World pipeline once.  The pipeline loads the
# checkpoint weights, tokenizer and auxiliary models.  We use
# bfloat16 precision by default, which is the tested precision for
# these models.  The ``load_prompt_refiner`` flag enables the prompt
# refiner, which improves short prompts but consumes additional
# memory.  Adjust it according to your GPU memory budget.
PIPELINE = Video2WorldPipeline.from_config(
    config=_config,
    device="cuda",
    torch_dtype=torch.bfloat16,
    load_prompt_refiner=True,
)


# ---------------------------------------------------------------------------
# FastAPI app
#
app = FastAPI(
    title="Cosmos‑Predict2 Video2World",
    version="0.1.0",
    summary=f"Serving Cosmos‑Predict2 Video2World {MODEL_SIZE} model",
)


class GenResponse(BaseModel):
    """Response model for the /generate endpoint."""
    video_path: str


@app.post("/generate", response_model=GenResponse)
async def generate(
    prompt: str = Form(..., description="Text prompt guiding the generation"),
    image: UploadFile = File(..., description="Image file to condition the model"),
    negative_prompt: str = Form(
        "",
        description=(
            "Negative text prompt to steer the model away from undesired content."
            " Leave blank to use the default negative prompt configured in the model."
        ),
    ),
    guidance: float = Form(
        7.0, description="Guidance scale controlling prompt adherence"
    ),
):
    """
    Generate a video conditioned on a text prompt and an input image.

    The uploaded image is temporarily persisted to disk because the
    Video2World pipeline expects a file path.  Generation is executed in
    a worker thread via ``asyncio.to_thread`` to avoid blocking the
    event loop.  On success, the path of the generated video is
    returned; you can download the file via the ``/download/{file_id}``
    endpoint.
    """
    # Validate content type early
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(422, "Expected an image/* upload")

    # Persist the uploaded image to a temporary file.  We cannot
    # directly pass a file‑like object to the pipeline.
    suffix = os.path.splitext(image.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await image.read())
        tmp_in.flush()
        input_path = tmp_in.name

    # Random seed for reproducibility; you could expose this as a query
    # parameter if deterministic outputs are desired.
    seed = torch.seed()

    def _run_generation() -> str:
        """Synchronous helper that runs the pipeline and writes the video."""
        # Call the pipeline.  See examples in the official script for
        # supported arguments【509323749675782†L348-L367】.  We use ``num_conditional_frames=1``
        # because we're conditioning on a single image.  To support
        # multi‑frame conditioning, you would need to upload a short
        # video and set this parameter to 5.
        video, _ = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio="16:9",
            input_path=input_path,
            num_conditional_frames=1,
            guidance=guidance,
            seed=seed,
            return_prompt=True,
        )
        # Determine fps from the model configuration.  The 2B and
        # 14B pipelines use ``state_t`` to set the video length; a
        # value of 24 corresponds to 16 fps and 20 corresponds to 10 fps.
        fps = 16 if PIPELINE.config.state_t == 24 else 10
        # Write the video to a file in OUT_DIR.  ``save_image_or_video``
        # automatically handles video tensors.
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", dir=OUT_DIR, delete=False
        ) as tmp_out:
            save_image_or_video(video, tmp_out.name, fps=fps)
            return os.path.basename(tmp_out.name)

    # Run the synchronous generation in a worker thread
    try:
        video_filename = await asyncio.to_thread(_run_generation)
    finally:
        # Clean up the temporary input file
        try:
            os.remove(input_path)
        except OSError:
            pass

    return {"video_path": video_filename}


@app.get("/download/{file_id}")
def download(file_id: str):
    """
    Download an already generated video.

    Videos are stored in ``OUT_DIR``.  The ``file_id`` should match the
    name returned by the ``/generate`` endpoint.  If the file does not
    exist a 404 error is returned.
    """
    path = os.path.join(OUT_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/ping")
def ping():
    """Simple health‑check endpoint."""
    return {
        "status": "ok",
        "model_size": MODEL_SIZE,
        "num_gpus": NUM_GPUS,
    }