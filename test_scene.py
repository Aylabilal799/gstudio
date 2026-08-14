import os
import sys
import logging
from videos import generate_genuine_ai_video

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A character walking in a dimly lit hallway"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "test_scene.mp4"

    print("=== Running Dedicated Low-Memory CogVideoX-2B Test ===")
    print(f"Prompt: {prompt}")
    print(f"Output File: {output_file}")

    try:
        res = generate_genuine_ai_video(prompt, output_file)
        print(f"\n✅ SUCCESS: Generated {res}")
        print(f"Dimensions: 1080x1920 (Vertical 9:16)")
    except Exception as e:
        print(f"\n❌ FAILURE [{e.__class__.__name__}]: {e}")
        sys.exit(1)
