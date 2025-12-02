#!/usr/bin/env python3
"""
Demo: Video File Upload Support for Cosmos Client

This demonstrates the new functionality that allows local video files 
to be used with the Cosmos video generation service.

NEW FEATURES:
1. Server now has an /upload_video endpoint
2. Client automatically uploads local video files when --video is used
3. Both single-frame and multi-frame conditioning now work with local videos

USAGE EXAMPLES:

# Single-frame conditioning (uses last frame of video)
python cosmos_client.py --prompt "Turn this into a cartoon" --video ./my_video.mp4 --num-frames 1

# Multi-frame conditioning (uses 5 frames from video)
python cosmos_client.py --prompt "Make it rain" --video ./my_video.mp4 --num-frames 5

# You can still use video URLs if preferred
python cosmos_client.py --prompt "Add snow" --video-url https://example.com/video.mp4 --num-frames 5

TECHNICAL DETAILS:
- When you use --video with a local file, the client automatically:
  1. Uploads the video to the server's /upload_video endpoint
  2. Gets a temporary URL back
  3. Uses that URL for video generation
- The server stores uploaded videos temporarily in /tmp/outputs/temp_uploads/
- Supports all common video formats: .mp4, .mov, .avi, .mkv, .webm
"""

def main():
    print(__doc__)
    
    print("\nTo test the new functionality:")
    print("1. Start the server: python cosmos_video_service.py")
    print("2. Use a local video file: python cosmos_client.py --prompt 'Make it magical' --video ./your_video.mp4 --num-frames 5")
    print("\nThe client will automatically upload your local video file and process it!")

if __name__ == "__main__":
    main()
