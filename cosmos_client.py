#!/usr/bin/env python3
"""
Cosmos Video Client
===================

A Python client for the Cosmos Video Generation Service that now supports

▸ **Images *and* Videos** as conditioning input  
▸ **num_conditional_frames** (1 = single-frame, ≥ 5 = multi-frame)  
▸ Both the **OpenAI-compatible** `/v1/chat/completions` and the
  **direct** `/generate` endpoints

--------------------------------------------------------------------
Quick examples
--------------------------------------------------------------------
# 1 — Single-frame from an image (default behaviour unchanged)
python cosmos_client.py --prompt "A robot in a garden" --image ./frame.jpg

# 2 — Single-frame from *last* frame of a video
python cosmos_client.py --prompt "Turn last frame into Pixar style" \
                        --video ./my_video.mp4 \
                        --num-frames 1

# 3 — Five-frame conditioning from a video
python cosmos_client.py --prompt "Make it snow" \
                        --video ./my_video.mp4 \
                        --num-frames 5

Run  `python cosmos_client.py -h`  to see all options.
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
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def health_check(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def ping(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/ping", timeout=10)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _encode_file_to_data_uri(path: str, mime_fallback: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            data = fh.read()
        ext = Path(path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
        }
        mime = mime_map.get(ext, mime_fallback)
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"

    # --------------------------------------------------------------------- #
    # Video upload helper
    # --------------------------------------------------------------------- #

    def upload_video(self, video_path: str) -> Dict[str, Any]:
        """Upload a local video file to the server and return a temporary URL."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                response = self.session.post(
                    f"{self.base_url}/upload_video",
                    files=files,
                    timeout=300  # Longer timeout for video uploads
                )
                response.raise_for_status()
                return response.json()
        except requests.RequestException as exc:
            return {
                "success": False,
                "error": f"Upload failed: {exc}",
                "details": getattr(exc.response, "text", ""),
            }

    # --------------------------------------------------------------------- #
    # OpenAI-compatible endpoint
    # --------------------------------------------------------------------- #

    def generate_video_openai_style(
        self,
        prompt: str,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        num_conditional_frames: int = 1,
        model: str = "cosmos-video-001",
    ) -> Dict[str, Any]:
        """
        Call `/v1/chat/completions`.

        Exactly one of {image_path, image_url, video_path, video_url}
        may be supplied.
        """
        # -----------------------------------------------------------------
        # Validate & prepare media
        # -----------------------------------------------------------------
        supplied = [v for v in (image_path, image_url, video_path, video_url) if v]
        if len(supplied) > 1:
            raise ValueError("Provide at most one media input (image or video).")

        content = [{"type": "text", "text": prompt}]
        if image_path:
            uri = self._encode_file_to_data_uri(image_path, "image/jpeg")
            content.append({"type": "image_url", "image_url": {"url": uri}})
        elif image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif video_path:
            # Upload local video file and get temporary URL
            print(f"Uploading video file: {video_path}")
            upload_result = self.upload_video(video_path)
            if not upload_result.get("success"):
                raise ValueError(f"Failed to upload video: {upload_result.get('error')}")
            
            video_url = f"{self.base_url}{upload_result['video_url']}"
            content.append({"type": "image_url", "image_url": {"url": video_url}})
        elif video_url:
            content.append({"type": "image_url", "image_url": {"url": video_url}})

        req_json = {
            "model": model,
            "num_conditional_frames": num_conditional_frames,
            "messages": [{"role": "user", "content": content}],
        }

        try:
            res = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=req_json,
                timeout=3000,
            )
            res.raise_for_status()
            payload = res.json()

            url = payload["choices"][0]["message"]["content"]
            meta = payload["choices"][0]["message"].get("metadata", {})
            full = f"{self.base_url}{url}" if url.startswith("/") else url
            return {
                "success": True,
                "video_url": url,
                "full_url": full,
                "metadata": meta,
                "response_id": payload["id"],
                "model": payload["model"],
            }
        except requests.RequestException as exc:
            return {
                "success": False,
                "error": f"HTTP request failed: {exc}",
                "details": getattr(exc.response, "text", ""),
            }

    # --------------------------------------------------------------------- #
    # Direct endpoint
    # --------------------------------------------------------------------- #

    def generate_video_direct(
        self,
        prompt: str,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        num_conditional_frames: int = 1,
        model_size: Optional[str] = None,
        num_gpus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call `/generate`."""
        supplied = [v for v in (image_path, image_url, video_path, video_url) if v]
        if len(supplied) > 1:
            raise ValueError("Provide at most one media input (image or video).")

        data: Dict[str, Any] = {
            "prompt": prompt,
            "num_conditional_frames": num_conditional_frames,
        }
        if model_size:
            data["model_size"] = model_size
        if num_gpus:
            data["num_gpus"] = num_gpus

        if image_path:
            data["image_url"] = self._encode_file_to_data_uri(image_path, "image/jpeg")
        elif image_url:
            data["image_url"] = image_url
        elif video_path:
            # Upload local video file and get temporary URL
            print(f"Uploading video file: {video_path}")
            upload_result = self.upload_video(video_path)
            if not upload_result.get("success"):
                raise ValueError(f"Failed to upload video: {upload_result.get('error')}")
            
            video_url = f"{self.base_url}{upload_result['video_url']}"
            data["image_url"] = video_url
        elif video_url:
            data["image_url"] = video_url

        try:
            res = self.session.post(
                f"{self.base_url}/generate", json=data, timeout=3000
            )
            res.raise_for_status()
            payload = res.json()
            if payload.get("success") and payload.get("download_url"):
                payload["full_download_url"] = (
                    f"{self.base_url}{payload['download_url']}"
                )
            return payload
        except requests.RequestException as exc:
            return {
                "success": False,
                "error": f"HTTP request failed: {exc}",
                "details": getattr(exc.response, "text", ""),
            }

    # --------------------------------------------------------------------- #
    # Download helper
    # --------------------------------------------------------------------- #

    def download_video(self, video_url: str, save_path: str) -> bool:
        if video_url.startswith("/"):
            video_url = f"{self.base_url}{video_url}"
        try:
            res = self.session.get(video_url, stream=True, timeout=300)
            res.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as fh:
                for chunk in res.iter_content(chunk_size=8192):
                    fh.write(chunk)
            print(f"✓ Downloaded to {save_path}")
            return True
        except Exception as exc:
            print(f"✗ Download failed: {exc}")
            return False


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #


def create_example_image() -> str:
    from PIL import Image, ImageDraw

    path = "/tmp/cosmos_example_input.jpg"
    img = Image.new("RGB", (512, 512), "lightblue")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill="white", outline="black", width=3)
    draw.ellipse([150, 150, 350, 350], fill="yellow", outline="orange", width=2)
    draw.text((200, 240), "TEST", fill="black")
    img.save(path, "JPEG", quality=95)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description="Cosmos Video Generation Client")
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument("--prompt", default="A robot picking up an apple from a table")

    media = p.add_mutually_exclusive_group()
    media.add_argument("--image", help="Path to input image file")
    media.add_argument("--image-url", help="URL to input image")
    media.add_argument("--video", help="Path to input video file")
    media.add_argument("--video-url", help="URL to input video")

    p.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="num_conditional_frames (1 = single-frame, ≥5 = multi-frame)",
    )
    p.add_argument(
        "--method",
        choices=["openai", "direct"],
        default="openai",
        help="API method to use",
    )
    p.add_argument("--model-size", help="Cosmos model size (direct method)")
    p.add_argument("--num-gpus", type=int, help="GPUs (direct method)")
    p.add_argument("--output", default="/tmp/generated_video.mp4")
    p.add_argument("--check-health", action="store_true", help="Only check health")
    p.add_argument("--create-example", action="store_true")

    args = p.parse_args()
    client = CosmosVideoClient(args.base_url)

    if args.check_health:
        print(client.health_check())
        return

    if args.create_example:
        args.image = create_example_image()
        print(f"Example image written to {args.image}")

    # --------------------------------------------------------------------- #
    # Ping service
    # --------------------------------------------------------------------- #
    try:
        info = client.ping()
        print(f"✓ Service OK – model {info.get('model')}")
    except Exception as exc:
        print(f"✗ Service unavailable: {exc}")
        sys.exit(1)

    # --------------------------------------------------------------------- #
    # Generate
    # --------------------------------------------------------------------- #
    print("Generating…")
    t0 = time.time()
    if args.method == "openai":
        res = client.generate_video_openai_style(
            prompt=args.prompt,
            image_path=args.image,
            image_url=args.image_url,
            video_path=args.video,
            video_url=args.video_url,
            num_conditional_frames=args.num_frames,
        )
        success = res.get("success")
        video_url = res.get("full_url")
    else:
        res = client.generate_video_direct(
            prompt=args.prompt,
            image_path=args.image,
            image_url=args.image_url,
            video_path=args.video,
            video_url=args.video_url,
            model_size=args.model_size,
            num_gpus=args.num_gpus,
            num_conditional_frames=args.num_frames,
        )
        success = res.get("success")
        video_url = res.get("full_download_url")

    if not success:
        print(f"✗ Generation failed: {res.get('error')}")
        if res.get("details"):
            print(res["details"])
        sys.exit(1)

    dt = time.time() - t0
    print(f"✓ Generated in {dt:.1f}s → {video_url}")

    # --------------------------------------------------------------------- #
    # Download
    # --------------------------------------------------------------------- #
    if video_url:
        client.download_video(video_url, args.output)


if __name__ == "__main__":
    main()
