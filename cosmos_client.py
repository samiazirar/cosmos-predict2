"""
Cosmos Video Client

A Python client for the Cosmos Video Generation Service that provides both
OpenAI-compatible and direct API interfaces.

Usage:
    python cosmos_client.py                           # Run with example
    python cosmos_client.py --help                    # Show all options
    python cosmos_client.py --prompt "your prompt"    # Custom prompt
"""

import os
import sys
import argparse
import base64
import requests
import time
from typing import Optional, Dict, Any
from pathlib import Path


class CosmosVideoClient:
    """Client for Cosmos Video Generation Service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the Cosmos video service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and get configuration."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Service health check failed: {e}")
    
    def ping(self) -> Dict[str, Any]:
        """Simple ping to check service availability."""
        try:
            response = self.session.get(f"{self.base_url}/ping")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Service ping failed: {e}")
    
    def _encode_image_to_data_uri(self, image_path: str) -> str:
        """Encode local image file to data URI."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Determine MIME type based on file extension
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        b64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{b64_data}"
    
    def generate_video_openai_style(
        self, 
        prompt: str, 
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        model: str = "cosmos-video-001"
    ) -> Dict[str, Any]:
        """Generate video using OpenAI-compatible chat completions endpoint.
        
        Args:
            prompt: Text prompt for video generation
            image_path: Path to local image file (optional)
            image_url: URL or data URI of image (optional) 
            model: Model name to use
            
        Returns:
            Dictionary with video URL and metadata
        """
        # Prepare message content
        content = [{"type": "text", "text": prompt}]
        
        # Add image if provided
        if image_path:
            image_uri = self._encode_image_to_data_uri(image_path)
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        elif image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        
        # Prepare request
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=3000  # 5 minutes timeout for video generation
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract video URL from response
            video_url = result["choices"][0]["message"]["content"]
            metadata = result["choices"][0]["message"].get("metadata", {})
            
            return {
                "success": True,
                "video_url": video_url,
                "full_url": f"{self.base_url}{video_url}" if video_url.startswith('/') else video_url,
                "metadata": metadata,
                "response_id": result["id"],
                "model": result["model"]
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {e}",
                "details": getattr(e.response, 'text', '') if hasattr(e, 'response') else ''
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {e}"
            }
    
    def generate_video_direct(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        model_size: Optional[str] = None,
        num_gpus: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate video using direct endpoint.
        
        Args:
            prompt: Text prompt for video generation
            image_path: Path to local image file (optional)
            image_url: URL or data URI of image (optional)
            model_size: Cosmos model size (optional)
            num_gpus: Number of GPUs to use (optional)
            
        Returns:
            Dictionary with video path and metadata
        """
        # Prepare image URL
        final_image_url = None
        if image_path:
            final_image_url = self._encode_image_to_data_uri(image_path)
        elif image_url:
            final_image_url = image_url
        
        # Prepare request data
        data = {"prompt": prompt}
        if final_image_url:
            data["image_url"] = final_image_url
        if model_size:
            data["model_size"] = model_size
        if num_gpus:
            data["num_gpus"] = num_gpus
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=data,
                timeout=3000  # 5 minutes timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Add full download URL
            if result.get("success") and result.get("download_url"):
                result["full_download_url"] = f"{self.base_url}{result['download_url']}"
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {e}",
                "details": getattr(e.response, 'text', '') if hasattr(e, 'response') else ''
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {e}"
            }
    
    def download_video(self, video_url: str, save_path: str) -> bool:
        """Download video from URL to local file.
        
        Args:
            video_url: Full URL to video file
            save_path: Local path to save the video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If it's a relative URL, make it absolute
            if video_url.startswith('/'):
                video_url = f"{self.base_url}{video_url}"
                
            response = self.session.get(video_url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Video downloaded successfully to: {save_path}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False


def create_example_image() -> str:
    """Create a simple example image for testing."""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some simple shapes
        draw.rectangle([100, 100, 400, 400], fill='white', outline='black', width=3)
        draw.ellipse([150, 150, 350, 350], fill='yellow', outline='orange', width=2)
        draw.text((200, 240), "TEST", fill='black')
        
        # Save to temp location
        example_path = "/tmp/cosmos_example_input.jpg"
        img.save(example_path, "JPEG", quality=95)
        return example_path
        
    except ImportError:
        print("PIL not available, skipping example image creation")
        return None


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Cosmos Video Generation Client")
    parser.add_argument("--base-url", default="http://localhost:8001", 
                       help="Base URL of the Cosmos service")
    parser.add_argument("--prompt", default="A robot picking up an apple from a table",
                       help="Text prompt for video generation")
    parser.add_argument("--image", help="Path to input image file")
    parser.add_argument("--image-url", help="URL to input image")
    parser.add_argument("--output", default="/tmp/generated_video.mp4",
                       help="Output path for downloaded video")
    parser.add_argument("--method", choices=["openai", "direct"], default="openai",
                       help="API method to use")
    parser.add_argument("--model-size", help="Cosmos model size (for direct method)")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs (for direct method)")
    parser.add_argument("--check-health", action="store_true",
                       help="Only check service health and exit")
    parser.add_argument("--create-example", action="store_true",
                       help="Create example image for testing")
    
    args = parser.parse_args()
    
    # Initialize client
    client = CosmosVideoClient(args.base_url)
    
    # Health check only
    if args.check_health:
        try:
            health = client.health_check()
            print("Service Health Check:")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Model Size: {health.get('cosmos_config', {}).get('model_size', 'unknown')}")
            print(f"  GPUs: {health.get('cosmos_config', {}).get('num_gpus', 'unknown')}")
            print(f"  Output Dir: {health.get('output_dir', 'unknown')}")
            return
        except Exception as e:
            print(f"Health check failed: {e}")
            sys.exit(1)
    
    # Create example image if requested
    if args.create_example:
        example_path = create_example_image()
        if example_path:
            print(f"Example image created at: {example_path}")
            args.image = example_path
        else:
            print("Could not create example image")
    
    print(f"Cosmos Video Client")
    print(f"Service URL: {args.base_url}")
    print(f"Method: {args.method}")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Image: {args.image}")
    if args.image_url:
        print(f"Image URL: {args.image_url}")
    print("-" * 50)
    
    # Check service health
    try:
        health = client.ping()
        print(f"✓ Service is running (Model: {health.get('model', 'unknown')})")
    except Exception as e:
        print(f"✗ Service not available: {e}")
        sys.exit(1)
    
    # Generate video
    print("Generating video...")
    start_time = time.time()
    
    if args.method == "openai":
        result = client.generate_video_openai_style(
            prompt=args.prompt,
            image_path=args.image,
            image_url=args.image_url
        )
    else:  # direct
        result = client.generate_video_direct(
            prompt=args.prompt,
            image_path=args.image,
            image_url=args.image_url,
            model_size=args.model_size,
            num_gpus=args.num_gpus
        )
    
    generation_time = time.time() - start_time
    
    if result["success"]:
        print(f"✓ Video generated successfully in {generation_time:.1f}s")
        
        # Get video URL
        if args.method == "openai":
            video_url = result["full_url"]
            print(f"Video URL: {video_url}")
            if result.get("metadata"):
                print(f"Metadata: {result['metadata']}")
        else:
            video_url = result["full_download_url"]
            print(f"Video URL: {video_url}")
            print(f"Video Path: {result.get('video_path', 'unknown')}")
        
        # Download video
        print(f"Downloading video to: {args.output}")
        if client.download_video(video_url, args.output):
            print("✓ Download completed successfully")
            print(f"Final video saved at: {args.output}")
        else:
            print("✗ Download failed")
            
    else:
        print(f"✗ Video generation failed: {result['error']}")
        if result.get('details'):
            print(f"Details: {result['details']}")


if __name__ == "__main__":
    main()
