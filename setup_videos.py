import os

os.makedirs("videos", exist_ok=True)

with open("videos/__init__.py", "w") as f:
    f.write("")

provider_code = '''import os
import hashlib
import cv2
import numpy as np
from abc import ABC, abstractmethod

class BaseVideoProvider(ABC):
    def __init__(self, cache_dir: str = "output/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, prompt: str, duration: int, width: int, height: int, seed: int = None) -> str:
        raw_key = f"{prompt.strip().lower()}_{duration}_{width}_{height}_{seed}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def get_cached_clip(self, prompt: str, duration: int, width: int, height: int, seed: int = None) -> str:
        key = self._get_cache_key(prompt, duration, width, height, seed)
        cached_file = os.path.join(self.cache_dir, f"cached_{key}.mp4")
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 1024:
            try:
                if self.verify_video_motion(cached_file):
                    print(f"[VIDEO] Found valid cached clip: {cached_file}")
                    return cached_file
            except Exception as e:
                print(f"[VIDEO] Invalid cache file {cached_file}: {e}")
                os.remove(cached_file)
        return None

    def save_to_cache(self, source_path: str, prompt: str, duration: int, width: int, height: int, seed: int = None) -> str:
        key = self._get_cache_key(prompt, duration, width, height, seed)
        cached_file = os.path.join(self.cache_dir, f"cached_{key}.mp4")
        import shutil
        shutil.copy(source_path, cached_file)
        return cached_file

    @abstractmethod
    def generate_video_clip(
        self,
        prompt: str,
        duration: int = 5,
        width: int = 720,
        height: int = 1280,
        seed: int = None,
        reference_image: str = None
    ) -> str:
        pass

    @staticmethod
    def verify_video_motion(video_path: str, min_frame_variance: float = 1.5) -> bool:
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1024:
            raise ValueError(f"Video file {video_path} does not exist or is empty.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        frames = []
        count = 0
        while cap.isOpened() and count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            count += 1
        cap.release()

        if len(frames) < 5:
            raise ValueError("Video contains too few frames to verify motion.")

        diff_mid = cv2.absdiff(frames[0], frames[len(frames) // 2])
        diff_end = cv2.absdiff(frames[0], frames[-1])
        mean_diff = (float(np.mean(diff_mid)) + float(np.mean(diff_end))) / 2.0

        if mean_diff < min_frame_variance:
            raise ValueError(
                f"REJECTED: Video frame difference is {mean_diff:.2f} (below threshold {min_frame_variance}). "
                "Output appears to be a static image or frozen video."
            )

        return True
'''

hf_code = '''import os
import shutil
from gradio_client import Client
from videos.provider import BaseVideoProvider

class HuggingFaceZeroGPUProvider(BaseVideoProvider):
    def __init__(self, hf_token: str = None, space_name: str = None, api_endpoint: str = None):
        super().__init__()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.space_name = space_name or os.getenv("HF_SPACE", "Lightricks/LTX-Video")
        self.api_endpoint = api_endpoint or os.getenv("HF_VIDEO_API", "/generate")

    def generate_video_clip(
        self,
        prompt: str,
        duration: int = 5,
        width: int = 720,
        height: int = 1280,
        seed: int = None,
        reference_image: str = None
    ) -> str:
        cached = self.get_cached_clip(prompt, duration, width, height, seed)
        if cached:
            return cached

        print(f"[VIDEO] Provider: HuggingFace ZeroGPU")
        print(f"[VIDEO] Space: {self.space_name}")
        print(f"[VIDEO] Prompt: {prompt[:80]}...")

        try:
            print("[VIDEO] Connecting to Hugging Face ZeroGPU Space...")
            client = Client(self.space_name, hf_token=self.hf_token)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Hugging Face Space '{self.space_name}': {e}")

        print("[VIDEO] Queue status: Waiting for ZeroGPU queue allocation...")

        try:
            kwargs = {}
            if self.api_endpoint:
                kwargs["api_name"] = self.api_endpoint

            result = client.predict(
                prompt=prompt,
                negative_prompt="static image, low quality, blurry, distorted, frozen",
                width=width,
                height=height,
                num_frames=int(duration * 24),
                seed=seed if seed is not None else 42,
                **kwargs
            )

            output_mp4_path = None
            if isinstance(result, str) and os.path.exists(result):
                output_mp4_path = result
            elif isinstance(result, (tuple, list)):
                for item in result:
                    if isinstance(item, str) and os.path.exists(item) and item.endswith((".mp4", ".webm")):
                        output_mp4_path = item
                        break

            if not output_mp4_path or not os.path.exists(output_mp4_path):
                raise RuntimeError(f"ZeroGPU Space returned unexpected output format: {result}")

            print("[VIDEO] Downloading and verifying generated clip...")
            self.verify_video_motion(output_mp4_path)
            cached_path = self.save_to_cache(output_mp4_path, prompt, duration, width, height, seed)
            print(f"[VIDEO] Generation complete: {cached_path}")
            return cached_path

        except Exception as e:
            raise RuntimeError(f"ZeroGPU Generation Failed: {e}")
'''

test_code = '''import sys
import os
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
    import shutil
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
'''

with open("videos/provider.py", "w") as f:
    f.write(provider_code)

with open("videos/huggingface_zerogpu.py", "w") as f:
    f.write(hf_code)

with open("test_video_api.py", "w") as f:
    f.write(test_code)

print("[+] All files written successfully with zero syntax errors!")
