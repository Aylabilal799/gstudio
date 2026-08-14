import os
import time
import requests
from videos.provider import BaseVideoProvider

class QwenVideoProvider(BaseVideoProvider):
    def generate_clip(self, prompt: str, duration: int = 5, aspect_ratio: str = "9:16", ref_image: str = None) -> str:
        url = f"{self.base_url.rstrip('/')}/services/aigc/text2video/video-synthesis"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }

        # Size mapping for 9:16 vertical short format
        size_map = {"9:16": "720*1280", "16:9": "1280*720", "1:1": "960*960"}
        size = size_map.get(aspect_ratio, "720*1280")

        payload = {
            "model": self.model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "size": size,
                "duration": duration
            }
        }

        if ref_image and os.path.exists(ref_image):
            # If reference image provided, pass reference URL if uploaded or configure I2V payload
            payload["input"]["img_url"] = ref_image

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"DashScope Video API submit error ({response.status_code}): {response.text}")

        res_data = response.json()
        task_id = res_data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Failed to obtain task_id from video generation response: {res_data}")

        # Task Polling
        task_url = f"{self.base_url.rstrip('/')}/tasks/{task_id}"
        poll_headers = {"Authorization": f"Bearer {self.api_key}"}

        max_polls = 60
        poll_interval = 10
        video_url = None

        for _ in range(max_polls):
            time.sleep(poll_interval)
            status_res = requests.get(task_url, headers=poll_headers, timeout=20)
            if status_res.status_code != 200:
                continue

            s_data = status_res.json()
            task_status = s_data.get("output", {}).get("task_status")

            if task_status == "SUCCEEDED":
                video_url = s_data.get("output", {}).get("video_url")
                break
            elif task_status in ["FAILED", "CANCELED"]:
                code = s_data.get("output", {}).get("code")
                msg = s_data.get("output", {}).get("message")
                raise RuntimeError(f"Video synthesis task {task_id} failed: [{code}] {msg}")

        if not video_url:
            raise TimeoutError(f"Video synthesis task {task_id} timed out after {max_polls * poll_interval} seconds.")

        # Download Video Clip
        clip_dir = "output/temp_clips"
        os.makedirs(clip_dir, exist_ok=True)
        local_filename = os.path.join(clip_dir, f"clip_{task_id}.mp4")

        clip_res = requests.get(video_url, stream=True, timeout=60)
        with open(local_filename, "wb") as f:
            for chunk in clip_res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Frame motion integrity check
        self.verify_video_motion(local_filename)
        return local_filename
