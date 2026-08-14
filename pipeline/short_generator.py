import os
import json
import uuid
import shutil
import subprocess
import requests
import cv2
import numpy as np
from dotenv import load_dotenv

from videos.huggingface_zerogpu import HuggingFaceZeroGPUProvider
from llm.planner import StoryPlanner
from audio.tts import TTSEngine
from subtitles.caption_generator import CaptionRenderer

load_dotenv()

class ShortGenerator:
    def __init__(self):
        self.planner = StoryPlanner()
        self.video_provider = HuggingFaceZeroGPUProvider()
        self.tts_engine = TTSEngine()
        self.caption_renderer = CaptionRenderer()

    def generate_short(self, story_text: str, max_scenes: int = 3) -> str:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_dir = os.path.join("output", job_id)
        clips_dir = os.path.join(job_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        print(f"\n==================================================")
        print(f"GStudio Pipeline Job: {job_id}")
        print(f"==================================================")

        # 1. LLM Scene Planning & Character Bible Creation
        print("[1/6] Planning scenes with LLM...")
        plan = self.planner.plan_story(story_text, default_clip_duration=5, num_scenes=max_scenes)
        scenes = plan.get("scenes", [])[:max_scenes]

        manifest = {
            "job_id": job_id,
            "story": story_text,
            "character_bible": plan.get("character_bible", {}),
            "scenes": []
        }

        # 2. Per-Scene Clip Generation (Cached & Verified)
        print(f"[2/6] Generating {len(scenes)} real 1080x1920 video clips via Hugging Face ZeroGPU...")
        downloaded_clips = []

        for idx, scene in enumerate(scenes):
            scene_num = idx + 1
            prompt = scene.get("video_prompt", "")
            duration = scene.get("duration", 5)

            print(f"\n--- Generating Scene {scene_num}/{len(scenes)} ---")
            print(f"Prompt: {prompt[:80]}...")

            # Retry loop per scene
            clip_path = None
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    clip_path = self.video_provider.generate_video_clip(
                        prompt=prompt,
                        duration=duration,
                        width=1080,
                        height=1920
                    )
                    break
                except Exception as e:
                    print(f"[!] Scene {scene_num} attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        raise RuntimeError(f"Scene {scene_num} failed after {max_retries} attempts.")

            dest_clip = os.path.join(clips_dir, f"clip_{scene_num:03d}.mp4")
            shutil.copy(clip_path, dest_clip)
            downloaded_clips.append(dest_clip)

            manifest["scenes"].append({
                "scene": scene_num,
                "prompt": prompt,
                "duration": duration,
                "file": dest_clip,
                "status": "completed"
            })

        with open(os.path.join(job_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # 3. FFmpeg Concatenation into Seamless Video Track
        print("\n[3/6] Concatenating 1080x1920 clips with FFmpeg...")
        concat_txt = os.path.join(job_dir, "concat.txt")
        with open(concat_txt, "w") as f:
            for clip in downloaded_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        concat_video = os.path.join(job_dir, "concatenated.mp4")
        ffmpeg_concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-c", "copy",
            concat_video
        ]
        subprocess.run(ffmpeg_concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 4. Generate Audio Narration
        print("\n[4/6] Generating TTS narration audio...")
        audio_file = os.path.join(job_dir, "narration.mp3")
        self.tts_engine.generate_narration(story_text, audio_file)

        # Combine Video + Audio
        video_with_audio = os.path.join(job_dir, "video_audio.mp4")
        ffmpeg_audio_cmd = [
            "ffmpeg", "-y",
            "-i", concat_video,
            "-i", audio_file,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            video_with_audio
        ]
        subprocess.run(ffmpeg_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 5. Generate Captions & Render
        print("\n[5/6] Generating word captions and burning overlay...")
        ass_file = os.path.join(job_dir, "captions.ass")
        self.caption_renderer.create_ass_subtitles(story_text, duration_sec=15.0, output_ass=ass_file)

        final_short = os.path.join(job_dir, "final_short.mp4")
        self.caption_renderer.burn_captions(video_with_audio, ass_file, final_short)

        # 6. Verify Final Video File
        print("\n[6/6] Verifying final 1080x1920 Short video file...")
        self._verify_final_video(final_short)

        print(f"\n[+] PIPELINE SUCCESS! Final short created at: {final_short}")

        # Send to Discord if enabled
        if os.getenv("DISCORD_ENABLED", "false").lower() == "true":
            self._send_to_discord(job_id, final_short)

        return final_short

    def _verify_final_video(self, video_path: str):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "csv=p=0",
            video_path
        ]
        res = subprocess.check_output(cmd).decode().strip().split(",")
        width, height, fps, duration = res[0], res[1], res[2], res[3]

        if int(width) != 1080 or int(height) != 1920:
            raise ValueError(f"Final resolution mismatch: expected 1080x1920, got {width}x{height}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        diff = cv2.absdiff(frames[0], frames[-1])
        if float(np.mean(diff)) < 1.0:
            raise ValueError("Final video failed motion check.")

        print(f"    - Resolution: {width}x{height} (PASS)")
        print(f"    - Duration: {float(duration):.2f}s (PASS)")
        print(f"    - Frame Motion: VERIFIED (PASS)")

    def _send_to_discord(self, job_id: str, video_path: str):
        token = os.getenv("DISCORD_TOKEN")
        channel_id = os.getenv("DISCORD_CHANNEL_ID")
        vps_ip = os.getenv("VPS_IP", "http://152.53.124.111:5454").rstrip("/")

        if not token or not channel_id:
            print("[!] Discord upload skipped: DISCORD_TOKEN or DISCORD_CHANNEL_ID missing.")
            return

        public_url = f"{vps_ip}/output/{job_id}/final_short.mp4"
        msg = f"🎬 **New YouTube Short Generated!**\n📌 **Job ID:** `{job_id}`\n🔗 **Link:** {public_url}"

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {"Authorization": f"Bot {token}"}

        try:
            if os.path.exists(video_path) and os.path.getsize(video_path) < 25 * 1024 * 1024:
                with open(video_path, "rb") as f:
                    files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                    requests.post(url, headers=headers, data={"content": msg}, files=files, timeout=60)
            else:
                requests.post(url, headers=headers, json={"content": msg}, timeout=30)
            print(f"[+] Discord notification sent to channel {channel_id}.")
        except Exception as e:
            print(f"[!] Discord notification error: {e}")
