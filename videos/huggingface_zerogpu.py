import os
import shutil
import subprocess
import cv2
import numpy as np
from gradio_client import Client
from videos.provider import BaseVideoProvider
from videos.huggingface_discovery import SpaceDiscoverer

class HuggingFaceZeroGPUProvider(BaseVideoProvider):
    def __init__(self, hf_token: str = None, space_name: str = None, api_endpoint: str = None):
        super().__init__()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.configured_space = space_name or os.getenv("HF_SPACE", "Lightricks/ltx-video-distilled")
        self.configured_endpoint = api_endpoint or os.getenv("HF_VIDEO_API", "/text_to_video")
        self.discoverer = SpaceDiscoverer(self.hf_token)

    def _get_active_client(self):
        if self.configured_space:
            print(f"[VIDEO] Using configured HF_SPACE: '{self.configured_space}'...")
            try:
                client = Client(self.configured_space, token=self.hf_token) if self.hf_token else Client(self.configured_space)
                return client, self.configured_space, self.configured_endpoint, True
            except Exception as e:
                print(f"[VIDEO] Error connecting to configured Space '{self.configured_space}': {e}")

        # Fallback to dynamic discovery
        candidates = self.discoverer.discover_spaces(max_inspect=10)
        valid = [c for c in candidates if c.gradio_available and c.usable_endpoint and not c.rejection_reason]

        if not valid:
            raise RuntimeError("No usable free public Hugging Face video generation Space currently available.")

        selected = valid[0]
        print(f"[VIDEO] Selected Candidate Space: '{selected.space_id}' (Endpoint: {selected.usable_endpoint})")
        client = Client(selected.space_id, token=self.hf_token) if self.hf_token else Client(selected.space_id)
        return client, selected.space_id, selected.usable_endpoint, selected.zerogpu

    def _convert_to_1080x1920(self, input_mp4: str, output_mp4: str):
        """Scales and pads video to exact 1080x1920 @ 30 FPS H.264 without stretching."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_mp4,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
            output_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def generate_video_clip(self, prompt: str, duration: int = 5, width: int = 1080, height: int = 1920, seed: int = None, reference_image: str = None) -> str:
        cached = self.get_cached_clip(prompt, duration, width, height, seed)
        if cached:
            return cached

        client, space_id, endpoint, zerogpu_status = self._get_active_client()

        print(f"[VIDEO] Provider: HuggingFace ZeroGPU")
        print(f"[VIDEO] Active Space: {space_id}")
        print(f"[VIDEO] Endpoint: {endpoint}")
        print(f"[VIDEO] Prompt: {prompt[:80]}...")
        print("[VIDEO] Requesting highest supported portrait resolution (512x704) from model...")

        try:
            # Request 512x704 portrait format natively from LTX model
            if "ltx-video" in space_id.lower() or endpoint == "/text_to_video":
                result = client.predict(
                    prompt=prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    input_image_filepath=None,
                    input_video_filepath=None,
                    height_ui=704.0,   # Portrait Height
                    width_ui=512.0,    # Portrait Width
                    mode="text-to-video",
                    duration_ui=float(duration),
                    ui_frames_to_use=float(duration * 12),
                    seed_ui=seed if seed is not None else 42,
                    randomize_seed=True,
                    ui_guidance_scale=1.0,
                    improve_texture_flag=True,
                    api_name=endpoint
                )
            else:
                kwargs = {"api_name": endpoint} if endpoint else {}
                result = client.predict(prompt=prompt, **kwargs)

            # Extract raw output file
            raw_mp4_path = None
            if isinstance(result, str) and os.path.exists(result):
                raw_mp4_path = result
            elif isinstance(result, (tuple, list)):
                for item in result:
                    if isinstance(item, str) and os.path.exists(item) and item.endswith((".mp4", ".webm")):
                        raw_mp4_path = item
                        break
                    elif isinstance(item, dict) and "video" in item:
                        raw_mp4_path = item.get("video")
                        break
            elif isinstance(result, dict):
                raw_mp4_path = result.get("video") or result.get("path")

            if not raw_mp4_path or not os.path.exists(raw_mp4_path):
                raise RuntimeError(f"Space returned unexpected output format: {result}")

            print("[VIDEO] Validating native video motion...")
            self.verify_video_motion(raw_mp4_path)

            # Convert to 1080x1920 Short format via FFmpeg
            os.makedirs("output/converted", exist_ok=True)
            converted_1080x1920_path = os.path.join("output/converted", f"converted_{os.path.basename(raw_mp4_path)}")

            print("[VIDEO] Converting native clip to 1080x1920 (9:16 portrait) via FFmpeg...")
            self._convert_to_1080x1920(raw_mp4_path, converted_1080x1920_path)

            # Verify final converted video motion
            self.verify_video_motion(converted_1080x1920_path)

            cached_path = self.save_to_cache(converted_1080x1920_path, prompt, duration, width, height, seed)
            print(f"[VIDEO] Final 1080x1920 Short clip complete: {cached_path}")
            return cached_path

        except Exception as e:
            raise RuntimeError(f"Video Generation Failed on '{space_id}': {e}")
