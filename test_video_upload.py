#!/usr/bin/env python3
"""
Test script to demonstrate the new video upload functionality.
"""

import os
import sys
from cosmos_client import CosmosVideoClient

def test_video_upload():
    """Test uploading a local video file."""
    client = CosmosVideoClient("http://localhost:8001")
    
    # Test with a hypothetical video file
    video_path = "/path/to/your/video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable to point to an actual video file.")
        return
    
    try:
        # Test upload
        print(f"Testing video upload: {video_path}")
        upload_result = client.upload_video(video_path)
        
        if upload_result.get("success"):
            print(f"✓ Upload successful!")
            print(f"  Video URL: {upload_result['video_url']}")
            print(f"  File ID: {upload_result['file_id']}")
            print(f"  Size: {upload_result['size']} bytes")
            
            # Test generation with uploaded video
            print("\nTesting video generation with multi-frame conditioning...")
            result = client.generate_video_openai_style(
                prompt="Make it snow in this scene",
                video_path=video_path,  # This should now work!
                num_conditional_frames=5
            )
            
            if result.get("success"):
                print(f"✓ Generation successful!")
                print(f"  Video URL: {result['full_url']}")
            else:
                print(f"✗ Generation failed: {result.get('error')}")
        else:
            print(f"✗ Upload failed: {upload_result.get('error')}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_video_upload()
