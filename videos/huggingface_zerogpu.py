import os
import shutil
from gradio_client import Client
from videos.provider import BaseVideoProvider

CANDIDATE_SPACES = [
    "fffiloni/LTX-Video",
    "multimodalart/LTX-Video-ZeroGPU",
    "AIDC-AI/Wan2.1-T2V-Turbo",
    "TencentARC/HunyuanVideo"
]

class HuggingFaceZeroGPUProvider(BaseVideoProvider):
    def __init__(self, hf_token: str = None, space_name: str = None, api_endpoint: str = None):
        super().__init__()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.space_name = space_name or os.getenv("HF_SPACE", "fffiloni/LTX-Video")
        self.api_endpoint = api_endpoint or os.getenv("HF_VIDEO_API")

    def _connect_client(self):
        spaces_to_try = [self.space_name] + [s for s in CANDIDATE_SPACES if s != self.space_name]

        for space in spaces_to_try:
            print(f"[VIDEO] Connecting to Hugging Face Space: '{space}'...")
            try:
                if self.hf_token:
                    client = Client(space, token=self.hf_token)
                else:
                    client = Client(space)
                print(f"[VIDEO] Successfully connected to Space: '{space}'")
                return client, space
            except Exception as e:
                print(f"[VIDEO] Space '{space}' connection failed: {e}")
                continue

        raise RuntimeError("Could not connect to any public Hugging Face ZeroGPU Video Spaces.")

    def generate_video_clip(self, prompt: str, duration: int = 5, width: int = 720, height: int = 1280, seed: int = None, reference_image: str = None) -> str:
        cached = self.get_cached_clip(prompt, duration, width, height, seed)
        if cached:
            return cached

        print(f"[VIDEO] Provider: HuggingFace ZeroGPU")
        print(f"[VIDEO] Prompt: {prompt[:80]}...")

        client, active_space = self._connect_client()
        print(f"[VIDEO] Active Space: {active_space}")
        print("[VIDEO] Queue status: Waiting for ZeroGPU queue allocation...")

        try:
            # Auto-detect available API endpoints from Gradio Client
            api_info = client.view_api(return_format="dict")
            endpoints = list(api_info.get("named_endpoints", {}).keys()) + list(api_info.get("unnamed_endpoints", {}).keys())

            target_endpoint = self.api_endpoint
            if not target_endpoint:
                if "/generate" in endpoints:
                    target_endpoint = "/generate"
                elif "/predict" in endpoints:
                    target_endpoint = "/predict"
                elif len(endpoints) > 0:
                    target_endpoint = endpoints[0]
                else:
                    target_endpoint = None

            print(f"[VIDEO] Calling endpoint: {target_endpoint or 'default'}")

            # Prepare parameters
            predict_kwargs = {"prompt": prompt}
            if target_endpoint:
                predict_kwargs["api_name"] = target_endpoint

            result = client.predict(**predict_kwargs)

            output_mp4_path = None
            if isinstance(result, str) and os.path.exists(result):
                output_mp4_path = result
            elif isinstance(result, (tuple, list)):
                for item in result:
                    if isinstance(item, str) and os.path.exists(item) and item.endswith((".mp4", ".webm")):
                        output_mp4_path = item
                        break
            elif isinstance(result, dict):
                output_mp4_path = result.get("video") or result.get("path")

            if not output_mp4_path or not os.path.exists(output_mp4_path):
                raise RuntimeError(f"ZeroGPU Space returned unexpected output format: {result}")

            print("[VIDEO] Downloading and verifying generated clip...")
            self.verify_video_motion(output_mp4_path)
            cached_path = self.save_to_cache(output_mp4_path, prompt, duration, width, height, seed)
            print(f"[VIDEO] Generation complete: {cached_path}")
            return cached_path

        except Exception as e:
            raise RuntimeError(f"ZeroGPU Generation Failed on '{active_space}': {e}")
