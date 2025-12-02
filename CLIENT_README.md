# Cosmos Video Client

A Python client for the Cosmos Video Generation Service that provides both OpenAI-compatible and direct API interfaces for generating videos from text prompts and optional images.

## Features

- 🎬 **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions endpoint
- 🚀 **Direct API**: More control over Cosmos-specific parameters
- 🖼️ **Image Support**: Generate videos from text + image inputs
- 📥 **Easy Downloads**: Automatic video downloading with progress tracking
- 🔧 **Health Monitoring**: Built-in service health checks
- 📝 **Multiple Examples**: Command-line, programmatic, and Jupyter notebook examples

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-client.txt
```

Or manually:
```bash
pip install requests pillow
```

### 2. Start the Cosmos Service

Make sure your Cosmos video service is running:
```bash
uvicorn cosmos_video_service:app --port 8001
```

### 3. Run the Example

```bash
# Quick example with default settings
python example_usage.py

# Or use the command-line client
python cosmos_client.py --prompt "A robot picking up an apple"
```

## Usage Examples

### Command Line Interface

```bash
# Basic usage
python cosmos_client.py --prompt "A cat walking on a table"

# With custom image
python cosmos_client.py --prompt "The robot moves" --image /path/to/image.jpg

# Check service health
python cosmos_client.py --check-health

# Create example image for testing
python cosmos_client.py --create-example --prompt "Test scene"

# Use direct API with custom parameters
python cosmos_client.py --method direct --model-size 14B --num-gpus 1
```

### Programmatic Usage

```python
from cosmos_client import CosmosVideoClient

# Initialize client
client = CosmosVideoClient("http://localhost:8001")

# Check service health
health = client.health_check()
print(f"Service status: {health['status']}")

# Generate video (OpenAI-style)
result = client.generate_video_openai_style(
    prompt="A robot arm picking up a red apple",
    image_path="/path/to/image.jpg"  # optional
)

if result["success"]:
    print(f"Video URL: {result['video_url']}")
    
    # Download the video
    client.download_video(result['full_url'], "/tmp/my_video.mp4")
else:
    print(f"Error: {result['error']}")
```

### Using with Image Input

```python
# From local file
result = client.generate_video_openai_style(
    prompt="The scene comes to life",
    image_path="/path/to/input.jpg"
)

# From URL
result = client.generate_video_openai_style(
    prompt="The scene comes to life", 
    image_url="https://example.com/image.jpg"
)

# From data URI (base64)
result = client.generate_video_openai_style(
    prompt="The scene comes to life",
    image_url="data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
)
```

### Direct API (More Control)

```python
# Use direct API for Cosmos-specific parameters
result = client.generate_video_direct(
    prompt="A butterfly flying in a garden",
    model_size="14B",
    num_gpus=1,
    image_path="/path/to/image.jpg"
)
```

## API Reference

### CosmosVideoClient

#### Constructor
```python
CosmosVideoClient(base_url: str = "http://localhost:8001")
```

#### Methods

**health_check()** → Dict[str, Any]
- Check service health and get configuration

**ping()** → Dict[str, Any]  
- Simple availability check

**generate_video_openai_style(prompt, image_path=None, image_url=None, model="cosmos-video-001")** → Dict[str, Any]
- Generate video using OpenAI-compatible endpoint
- Returns: `{"success": bool, "video_url": str, "metadata": dict, ...}`

**generate_video_direct(prompt, image_path=None, image_url=None, model_size=None, num_gpus=None)** → Dict[str, Any]
- Generate video using direct endpoint with more control
- Returns: `{"success": bool, "video_path": str, "download_url": str, ...}`

**download_video(video_url, save_path)** → bool
- Download video from URL to local file
- Returns: True if successful

## Examples Included

### 1. Command Line Tool (`cosmos_client.py`)
Full-featured CLI with all options:
```bash
python cosmos_client.py --help
```

### 2. Simple Example Script (`example_usage.py`)
Interactive script with predefined prompts:
```bash
python example_usage.py
```

### 3. Jupyter Notebook (`cosmos_client_example.ipynb`)
Complete interactive tutorial with:
- Service setup verification
- Text-only video generation
- Image + text video generation
- Direct API usage
- Video display in notebook

## Configuration

### Environment Variables

The client respects these environment variables:

- `COSMOS_SERVICE_URL`: Default service URL (default: `http://localhost:8001`)

### Service Configuration

The client automatically detects service configuration through health checks:

```python
health = client.health_check()
print(health['cosmos_config'])
# {
#   "model_size": "14B",
#   "num_gpus": 1,
#   "disable_guardrail": true,
#   "disable_prompt_refiner": true
# }
```

## Error Handling

The client provides detailed error information:

```python
result = client.generate_video_openai_style("test prompt")

if not result["success"]:
    print(f"Error: {result['error']}")
    if result.get('details'):
        print(f"Details: {result['details']}")
```

Common errors:
- **Service unavailable**: Check if cosmos_video_service is running
- **Invalid image**: Check image file path or URL
- **Generation timeout**: Video generation can take several minutes
- **Download failed**: Check output directory permissions

## Tips and Best Practices

### 1. Service Health Monitoring
Always check service health before generating videos:
```python
try:
    health = client.health_check()
    print("✓ Service ready")
except Exception as e:
    print(f"✗ Service unavailable: {e}")
```

### 2. Image Preparation
- Use high-quality images (512x512 or larger)
- JPEG/PNG formats work best
- Images are automatically converted to RGB

### 3. Prompt Engineering
- Be specific about desired actions
- Include object descriptions and movements
- Examples: "A robot arm slowly picks up the red apple" vs "robot apple"

### 4. Performance Considerations
- Video generation can take 2-5 minutes depending on hardware
- Use appropriate timeouts for your use case
- Consider caching generated videos

### 5. Batch Processing
```python
prompts = [
    "Scene 1: Robot approaches apple",
    "Scene 2: Robot grasps apple", 
    "Scene 3: Robot lifts apple"
]

videos = []
for prompt in prompts:
    result = client.generate_video_openai_style(prompt)
    if result["success"]:
        videos.append(result["full_url"])
```

## Troubleshooting

### Common Issues

**Service Connection Failed**
```bash
# Check if service is running
curl http://localhost:8001/health

# Start the service
uvicorn cosmos_video_service:app --port 8001
```

**Import Errors**
```bash
pip install -r requirements-client.txt
```

**Permission Denied (Downloads)**
```bash
# Make sure output directory exists and is writable
mkdir -p /tmp/cosmos_videos
chmod 755 /tmp/cosmos_videos
```

**Video Generation Timeout**
- Increase timeout in client code
- Check GPU availability
- Monitor service logs

### Debug Mode

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Client will now show detailed request/response info
```

## License

This client follows the same license as the Cosmos project.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Submit a pull request

---

For more information about the Cosmos video generation system, see the main project documentation.
