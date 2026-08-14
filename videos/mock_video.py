import os
import subprocess
from videos.provider import BaseVideoProvider

class MockVideoProvider(BaseVideoProvider):
    def generate_clip(self, prompt: str, duration: int = 5, aspect_ratio: str = "9:16", ref_image: str = None) -> str:
        os.makedirs("output/mock_clips", exist_ok=True)
        out_path = f"output/mock_clips/mock_{abs(hash(prompt))[:8]}.mp4"

        # Generate synthetic test clip using FFmpeg testsrc
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=720x1280:rate=30",
            "-vf", f"drawtext=text='{prompt[:20]}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
