# Cosmos Video Generator

This directory contains scripts for generating videos using Cosmos Predict2.

## Files

### `cosmos_video_generator.py`
The main isolated function for generating videos with Cosmos. This script wraps the torchrun command to invoke the `examples/video2world.py` script with proper parameters.

**Key Features:**
- Isolated function `generate_cosmos_video()` for programmatic use
- Simplified wrapper `generate_cosmos_video_simple()` for easy integration
- Proper error handling and validation
- Automatic output directory creation
- Support for all video2world.py parameters

**Usage:**
```python
from cosmos_video_generator import generate_cosmos_video

result = generate_cosmos_video(
    prompt="A beautiful sunset over mountains",
    input_path="path/to/input/image.jpg",
    model_size="14B",
    num_gpus=1
)

if result["success"]:
    print(f"Video generated: {result['output_path']}")
else:
    print(f"Error: {result['error']}")
```

**Command Line Usage:**
```bash
python cosmos_video_generator.py "Your prompt here" "input/image.jpg" "output/video.mp4"
```

### `cosmos_video_service.py`
A FastAPI service that provides an OpenAI-compatible endpoint for video generation using Cosmos Predict2. This replaces the dummy video service with real Cosmos generation.

**Features:**
- OpenAI-compatible `/v1/chat/completions` endpoint
- Accepts text prompts and image inputs
- Returns video URLs instead of text responses
- Direct generation endpoint at `/generate`
- Health check at `/health`

**Usage:**
```bash
uvicorn cosmos_video_service:app --port 8001
```

### `test_cosmos_generator.py`
Test suite for validating the Cosmos video generator functionality.

**Usage:**
```bash
python test_cosmos_generator.py
```

## Environment Variables

The service can be configured using environment variables:

- `COSMOS_MODEL_SIZE`: Model size to use ("2B" or "14B", default: "14B")
- `COSMOS_NUM_GPUS`: Number of GPUs to use (default: 1)
- `COSMOS_DISABLE_GUARDRAIL`: Disable guardrail checks ("true"/"false", default: "true")
- `COSMOS_DISABLE_PROMPT_REFINER`: Disable prompt refiner ("true"/"false", default: "true")
- `OUT_DIR`: Output directory for generated videos (default: "/tmp/outputs")

## Requirements

- Cosmos Predict2 environment properly set up
- Required Python packages: `fastapi`, `uvicorn`, `pillow`, `requests`
- Proper GPU setup for model inference
- Input assets in the `assets/video2world/` directory

## Command Structure

The generator executes commands in this format:
```bash
PYTHONPATH=. torchrun --nproc_per_node=${NUM_GPUS} examples/video2world.py \
    --model_size 14B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT}" \
    --save_path output/video2world_14b_${NUM_GPUS}gpu.mp4 \
    --num_gpus ${NUM_GPUS} \
    --disable_guardrail \
    --disable_prompt_refiner
```

## Error Handling

The generator includes comprehensive error handling for:
- Invalid input parameters
- Missing input files
- Command execution failures
- Timeouts (1 hour default)
- Environment setup issues

## Integration with Existing Services

The `cosmos_video_service.py` can be used as a drop-in replacement for `dummy_video_service.py` in existing applications that expect OpenAI-compatible video generation endpoints.
