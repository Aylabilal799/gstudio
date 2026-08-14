import os
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
