#!/usr/bin/env python3
"""
Simple example demonstrating how to use the Cosmos Video Client.

This script shows both OpenAI-compatible and direct API usage patterns.
"""

import os
import sys
from cosmos_client import CosmosVideoClient

def main():
    # Initialize client
    client = CosmosVideoClient("http://localhost:8004")
    
    print("🎬 Cosmos Video Client Example")
    print("=" * 50)
    
    # 1. Check service health
    try:
        health = client.health_check()
        print(f"✓ Service is healthy")
        print(f"  Model: {health['cosmos_config']['model_size']}")
        print(f"  GPUs: {health['cosmos_config']['num_gpus']}")
    except Exception as e:
        print(f"✗ Service unavailable: {e}")
        print("Make sure the service is running:")
        print("  uvicorn cosmos_video_service:app --port 8001")
        return
    
    print()
    
    # 2. Example prompts to try
    example_prompts = [
        "A robot arm picking up a red apple from a wooden table",
        "A cat walking across a kitchen counter",
        "Rain falling on a window while coffee steams in a mug",
        "A paper airplane flying through an office",
        "Leaves falling from a tree in autumn"
    ]
    
    print("📝 Example Prompts:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"  {i}. {prompt}")
    
    # Let user choose or use default
    try:
        choice = input(f"\nChoose a prompt (1-{len(example_prompts)}) or press Enter for default: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(example_prompts):
            selected_prompt = example_prompts[int(choice) - 1]
        else:
            selected_prompt = example_prompts[0]  # Default
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    print(f"\n🎯 Using prompt: {selected_prompt}")
    
    # 3. Generate video using OpenAI-style API
    print("\n🚀 Generating video...")
    print("(This may take a few minutes depending on your hardware)")
    
    try:
        result = client.generate_video_openai_style(
            prompt=selected_prompt,
            # Note: No image provided - service will use default or generate without image
        )
        
        if result["success"]:
            print("✓ Video generated successfully!")
            print(f"  Video URL: {result['video_url']}")
            print(f"  Model used: {result.get('model', 'unknown')}")
            
            # Download the video
            output_path = f"/tmp/cosmos_example_{result['response_id'][:8]}.mp4"
            print(f"\n📥 Downloading video to: {output_path}")
            
            if client.download_video(result['full_url'], output_path):
                print("✓ Download completed!")
                print(f"🎬 Video saved at: {output_path}")
                
                # Show some helpful info
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"📊 File size: {file_size:.1f} MB")
                
                print("\n🎉 Success! You can now:")
                print(f"  • Open the video: {output_path}")
                print(f"  • View in browser: {result['full_url']}")
                
            else:
                print("✗ Download failed")
        else:
            print(f"✗ Generation failed: {result['error']}")
            if result.get('details'):
                print(f"Details: {result['details']}")
                
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
