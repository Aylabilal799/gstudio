import os
import json
import uuid
import shutil
import subprocess
import requests
from dotenv import load_dotenv

from videos.huggingface_zerogpu import HuggingFaceZeroGPUProvider
from llm.planner import StoryPlanner

load_dotenv()

class ShortGenerator:
    def __init__(self):
        self.max_clips = int(os.getenv("MAX_CLIPS_PER_JOB", "15"))
        self.max_seconds = int(os.getenv("MAX_GENERATED_SECONDS_PER_JOB", "75"))

        self.planner = StoryPlanner()
        self.video_provider = HuggingFaceZeroGPUProvider()

    def generate_short(self, story_text: str, max_scenes: int = 3):
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_dir = os.path.join("output", job_id)
        clips_dir = os.path.join(job_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        print(f"[*] Starting GStudio Short Job: {job_id}")

        # Plan story with LLM
        plan = self.planner.plan_story(story_text, max_scenes=max_scenes)
        scenes = plan.get("scenes", [])[:self.max_clips]

        total_requested_seconds = sum(s.get("duration", 5) for s in scenes)
        print(f"Requested clips: {len(scenes)}")
        print(f"Requested generated seconds: {total_requested_seconds}s (Limit: {self.max_seconds}s)")

        if total_requested_seconds > self.max_seconds:
            raise ValueError(f"Job exceeds MAX_GENERATED_SECONDS_PER_JOB ({self.max_seconds}s).")

        downloaded_clips = []
        for idx, scene in enumerate(scenes):
            print(f"\n--- Scene {idx + 1}/{len(scenes)} ---")
            prompt = scene["video_prompt"]
            duration = scene.get("duration", 5)

            # Call HuggingFace ZeroGPU provider
            clip_path = self.video_provider.generate_video_clip(
                prompt=prompt,
                duration=duration,
                width=720,   # High-quality near-portrait base format
                height=1280
            )

            dest_path = os.path.join(clips_dir, f"clip_{idx + 1:03d}.mp4")
            shutil.copy(clip_path, dest_path)
            downloaded_clips.append(dest_path)

        # Concatenate and normalize with FFmpeg to 1080x1920 @ 30fps
        concat_txt = os.path.join(job_dir, "concat.txt")
        with open(concat_txt, "w") as f:
            for clip in downloaded_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        final_output = os.path.join(job_dir, "final_short.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
            final_output
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        print(f"\n[+] Final Short generated successfully: {final_output}")

        # Send to Discord if enabled
        if os.getenv("DISCORD_ENABLED", "false").lower() == "true":
            self._notify_discord(job_id, final_output)

        return final_output

    def _notify_discord(self, job_id: str, video_path: str):
        token = os.getenv("DISCORD_TOKEN")
        vps_ip = os.getenv("VPS_IP", "http://152.53.124.111:5454").rstrip("/")
        channel_id = os.getenv("DISCORD_CHANNEL_ID")

        if not token or not channel_id:
            print("[!] Discord notification skipped: DISCORD_TOKEN or DISCORD_CHANNEL_ID not set.")
            return

        public_url = f"{vps_ip}/{job_id}/final_short.mp4"
        msg = f"🎬 **YouTube Short Complete!**\n📌 **Job ID:** `{job_id}`\n🔗 **Link:** {public_url}"

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {"Authorization": f"Bot {token}"}

        if os.path.exists(video_path) and os.path.getsize(video_path) < 25 * 1024 * 1024:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                payload = {"content": msg}
                requests.post(url, headers=headers, data=payload, files=files, timeout=60)
        else:
            requests.post(url, headers=headers, json={"content": msg}, timeout=30)
        print("[+] Discord notification dispatched.")
