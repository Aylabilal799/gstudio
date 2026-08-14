import sys
import os
import shutil
import subprocess
from dotenv import load_dotenv
from videos.huggingface_zerogpu import HuggingFaceZeroGPUProvider

load_dotenv()

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "A young man running toward a city bus that is leaving a modern urban bus stop, "
        "cinematic realistic video, natural human movement"
    )

    print("==================================================")
    print("Testing Hugging Face ZeroGPU Video Generation")
    print("==================================================")

    provider = HuggingFaceZeroGPUProvider()
    clip_path = provider.generate_video_clip(
        prompt=prompt,
        duration=5,
        width=720,
        height=1280
    )

    target_file = "test_video.mp4"
    shutil.copy(clip_path, target_file)

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "csv=p=0",
        target_file
    ]
    res = subprocess.check_output(cmd).decode().strip().split(",")
    width, height, fps_raw, duration = res[0], res[1], res[2], res[3]
    file_size_mb = os.path.getsize(target_file) / (1024 * 1024)

    provider.verify_video_motion(target_file)

    print("\nVIDEO SUCCESS")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {float(duration):.2f}s")
    print(f"FPS: {fps_raw}")
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
