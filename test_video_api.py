import sys
import os
import shutil
import subprocess
import cv2
import numpy as np
from dotenv import load_dotenv
from videos.huggingface_zerogpu import HuggingFaceZeroGPUProvider

load_dotenv()

def probe_video(video_path: str):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "csv=p=0",
        video_path
    ]
    res = subprocess.check_output(cmd).decode().strip().split(",")
    width, height, fps_raw, duration = res[0], res[1], res[2], res[3]
    return width, height, fps_raw, duration

def check_motion(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) < 5:
        return False, 0.0

    diff1 = cv2.absdiff(frames[0], frames[len(frames) // 2])
    diff2 = cv2.absdiff(frames[0], frames[-1])
    mean_diff = (float(np.mean(diff1)) + float(np.mean(diff2))) / 2.0
    return mean_diff >= 1.5, mean_diff

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "A young man running toward a city bus that is leaving a modern urban bus stop, "
        "cinematic realistic video, natural human movement"
    )

    print("==================================================")
    print("Testing Hugging Face ZeroGPU 1080x1920 Short Clip")
    print("==================================================")

    provider = HuggingFaceZeroGPUProvider()
    clip_path = provider.generate_video_clip(
        prompt=prompt,
        duration=5,
        width=1080,
        height=1920
    )

    target_file = "test_video_1080x1920.mp4"
    shutil.copy(clip_path, target_file)

    width, height, fps_raw, duration = probe_video(target_file)
    has_motion, motion_diff = check_motion(target_file)

    print("\n==================================================")
    print("ONE-CLIP TEST REPORT")
    print("==================================================")
    print(f"MODEL: LTX-Video Distilled")
    print(f"SPACE: Lightricks/ltx-video-distilled")
    print(f"NATIVE RESOLUTION: 512x704 (Portrait)")
    print(f"DURATION: {float(duration):.2f}s")
    print(f"FPS: {fps_raw}")
    print(f"FINAL RESOLUTION: {width}x{height}")
    print(f"ACTUAL MOTION: {'YES' if has_motion else 'NO'} (Variance score: {motion_diff:.2f})")
    print("==================================================")

if __name__ == "__main__":
    main()
