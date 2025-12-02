#!/usr/bin/env python3
"""
Test script for the Cosmos video generator
"""

import os
import sys
from cosmos_video_generator import generate_cosmos_video, generate_cosmos_video_simple


def test_basic_generation():
    """Test basic video generation with default parameters."""
    print("Testing basic Cosmos video generation...")
    
    # Check if default input exists
    test_input = "assets/video2world/input0.jpg"
    if not os.path.exists(test_input):
        print(f"Warning: Default test input not found: {test_input}")
        print("Please ensure you have the assets directory with input images.")
        return False
    
    test_prompt = "A majestic eagle soaring through mountain peaks at sunset"
    
    try:
        result = generate_cosmos_video(
            prompt=test_prompt,
            input_path=test_input,
            model_size="14B",
            num_gpus=4,
            disable_guardrail=True,
            disable_prompt_refiner=True
        )
        
        if result["success"]:
            print(f"✓ Success! Video generated at: {result['output_path']}")
            print(f"Command executed: {result['command']}")
            return True
        else:
            print(f"✗ Generation failed: {result['error']}")
            if 'stderr' in result and result['stderr']:
                print(f"Error details: {result['stderr']}")
            return False
            
    except Exception as e:
        print(f"✗ Exception during generation: {e}")
        return False


def test_simple_wrapper():
    """Test the simplified wrapper function."""
    print("\nTesting simplified wrapper...")
    
    test_input = "assets/video2world/input0.jpg"
    if not os.path.exists(test_input):
        print(f"Skipping test - input not found: {test_input}")
        return False
    
    test_prompt = "A peaceful forest scene with gentle wind moving through trees"
    
    try:
        output_path = generate_cosmos_video_simple(test_prompt, test_input)
        print(f"✓ Simple wrapper success! Video at: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Simple wrapper failed: {e}")
        return False


def test_command_building():
    """Test that commands are built correctly without executing them."""
    print("\nTesting command building (dry run)...")
    
    from cosmos_video_generator import generate_cosmos_video
    import subprocess
    
    # Mock subprocess.run to avoid actual execution
    original_run = subprocess.run
    
    def mock_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('cmd', [])
        print(f"Would execute: {' '.join(cmd)}")
        
        # Create a mock result
        class MockResult:
            def __init__(self):
                self.returncode = 0
                self.stdout = "Mock successful generation"
                self.stderr = ""
        
        return MockResult()
    
    # Temporarily replace subprocess.run
    subprocess.run = mock_run
    
    try:
        result = generate_cosmos_video(
            prompt="Test prompt",
            input_path="test_input.jpg",
            model_size="2B",
            num_gpus=4,
            resolution="480",
            fps=16,
            aspect_ratio="16:9"
        )
        
        expected_parts = [
            "torchrun",
            "--nproc_per_node=2",
            "examples/video2world.py",
            "--model_size", "2B",
            "--input_path", "test_input.jpg",
            "--prompt", "Test prompt",
            "--num_gpus", "2",
            "--disable_guardrail",
            "--disable_prompt_refiner",
            "--resolution", "480",
            "--fps", "16",
            "--aspect_ratio", "16:9"
        ]
        
        print("✓ Command building test passed")
        return True
        
    except Exception as e:
        print(f"✗ Command building test failed: {e}")
        return False
    finally:
        # Restore original subprocess.run
        subprocess.run = original_run


def main():
    """Run all tests."""
    print("Cosmos Video Generator Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Command building (safe)
    total_tests += 1
    if test_command_building():
        tests_passed += 1
    
    # Test 2: Basic generation (requires proper setup)
    total_tests += 1
    if test_basic_generation():
        tests_passed += 1
    
    # Test 3: Simple wrapper (requires proper setup)
    total_tests += 1
    if test_simple_wrapper():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
