#!/usr/bin/env python3
"""
Cosmos Video Generator

A module that provides an isolated function for generating videos using Cosmos Predict2.
This wraps the torchrun command to invoke video generation with the video2world.py example.
"""

import os
import subprocess
import tempfile
import uuid
from typing import Optional, Dict, Any
from pathlib import Path


def generate_cosmos_video(
    prompt: str,
    input_path: str,
    save_path: Optional[str] = None,
    model_size: str = "14B",
    num_gpus: int = 1,
    disable_guardrail: bool = True,
    disable_prompt_refiner: bool = True,
    capture_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a video using Cosmos Predict2 via torchrun.
    
    Args:
        prompt (str): Text prompt for video generation
        input_path (str): Path to input image or video for conditioning
        save_path (str, optional): Path to save the generated video. If None, generates a unique filename
        model_size (str): Size of the model to use ("2B" or "14B")
        num_gpus (int): Number of GPUs to use for generation
        disable_guardrail (bool): Whether to disable guardrail checks
        disable_prompt_refiner (bool): Whether to disable prompt refiner
        capture_output (bool): Whether to capture stdout/stderr. If False, output streams directly to console (shows progress bars)
        **kwargs: Additional arguments to pass to the video2world.py script
        
    Returns:
        Dict containing:
            - success (bool): Whether generation was successful
            - output_path (str): Path to the generated video file
            - error (str, optional): Error message if generation failed
            - command (str): The actual command that was executed
    """
    
    # Validate inputs
    if not prompt.strip():
        return {
            "success": False,
            "error": "Prompt cannot be empty",
            "output_path": None,
            "command": None
        }
    
    if not os.path.exists(input_path):
        return {
            "success": False,
            "error": f"Input file does not exist: {input_path}",
            "output_path": None,
            "command": None
        }
    
    if model_size not in ["2B", "14B"]:
        return {
            "success": False,
            "error": f"Invalid model size: {model_size}. Must be '2B' or '14B'",
            "output_path": None,
            "command": None
        }
    
    # Generate output path if not provided
    if save_path is None:
        output_dir = os.path.join("output", "cosmos_generated")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"video2world_{model_size}_{num_gpus}gpu_{uuid.uuid4().hex[:8]}.mp4")
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Build the torchrun command
    cmd_parts = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "examples/video2world.py",
        "--model_size", model_size,
        "--input_path", input_path,
        "--prompt", prompt,
        "--save_path", save_path,
        "--num_gpus", str(num_gpus)
    ]
    
    # Add optional flags
    if disable_guardrail:
        cmd_parts.append("--disable_guardrail")
    
    if disable_prompt_refiner:
        cmd_parts.append("--disable_prompt_refiner")
    
    # Add any additional keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, bool) and value:
            cmd_parts.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd_parts.extend([f"--{key}", str(value)])
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Build the final command string for reference
    command_str = f"PYTHONPATH=. {' '.join(cmd_parts)}"
    
    try:
        # Execute the command
        print(f"Executing: {command_str}")
        if capture_output:
            result = subprocess.run(
                cmd_parts,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_path": save_path,
                    "command": command_str,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": f"Command failed with return code {result.returncode}",
                    "output_path": None,
                    "command": command_str,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            # Stream output directly to console (shows progress bars)
            result = subprocess.run(
                cmd_parts,
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_path": save_path,
                    "command": command_str
                }
            else:
                return {
                    "success": False,
                    "error": f"Command failed with return code {result.returncode}",
                    "output_path": None,
                    "command": command_str
                }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 1 hour",
            "output_path": None,
            "command": command_str
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "output_path": None,
            "command": command_str
        }


def generate_cosmos_video_simple(prompt: str, input_path: str, output_path: str = None) -> str:
    """
    Simplified wrapper that returns just the output path or raises an exception.
    
    Args:
        prompt (str): Text prompt for video generation
        input_path (str): Path to input image or video for conditioning
        output_path (str, optional): Path to save the generated video
        
    Returns:
        str: Path to the generated video file
        
    Raises:
        RuntimeError: If video generation fails
    """
    result = generate_cosmos_video(prompt, input_path, output_path)
    
    if result["success"]:
        return result["output_path"]
    else:
        raise RuntimeError(f"Video generation failed: {result['error']}")


# Example usage and testing functions
def test_generation():
    """Test function to verify the generator works."""
    # This would need an actual input image/video to work
    test_input = "assets/video2world/input0.jpg"
    test_prompt = "A beautiful sunset over the mountains"
    
    if os.path.exists(test_input):
        print("Testing Cosmos video generation...")
        result = generate_cosmos_video(
            prompt=test_prompt,
            input_path=test_input,
            model_size="14B",
            num_gpus=1
        )
        
        if result["success"]:
            print(f"✓ Generation successful! Video saved to: {result['output_path']}")
        else:
            print(f"✗ Generation failed: {result['error']}")
        
        return result
    else:
        print(f"Test input file not found: {test_input}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 2:
        prompt = sys.argv[1]
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"Generating video with prompt: '{prompt}'")
        print(f"Input: {input_path}")
        
        try:
            result_path = generate_cosmos_video_simple(prompt, input_path, output_path)
            print(f"✓ Video generated successfully: {result_path}")
        except RuntimeError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python cosmos_video_generator.py '<prompt>' '<input_path>' [output_path]")
        print("Running test...")
        test_generation()
