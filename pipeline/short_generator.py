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

    def generate_short(self, story_text: str, max_scenes: int = 4, progress_callback=None) -> str:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_dir = os.path.join("output", "jobs", job_id)
        clips_dir = os.path.join(job_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        def update_progress(msg: str):
            print(f"[PIPELINE] {msg}")
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception as e:
                    print(f"[PIPELINE] Progress callback error: {e}")

        update_progress("🎬 Starting video generation...")

        # 1. Master Audio Timeline First (Kokoro ONNX TTS)
        update_progress("🎙️ Generating female narration (Kokoro af_heart)...")
        audio_file = os.path.join(job_dir, "narration.wav")
        self.tts_engine.generate_narration(story_text, audio_file)

        audio_duration = self._get_audio_duration(audio_file)
        if audio_duration <= 0:
            raise RuntimeError("Generated narration audio has 0 duration.")

        print(f"[PIPELINE] Master Narration Duration: {audio_duration:.2f} seconds")

        # 2. Plan Story Beats Aligned to Narration Duration
        update_progress(f"🎬 Planning visual story beats for {audio_duration:.2f}s timeline...")
        plan = self.planner.plan_story_beats(story_text, total_narration_duration=audio_duration)
        beats = plan.get("beats", [])

        update_progress(f"✅ Visual story beats planned: {len(beats)}")

        manifest = {
            "job_id": job_id,
            "story": story_text,
            "master_narration_duration": audio_duration,
            "character_bible": plan.get("character_bible", {}),
            "beats": []
        }

        # 3. Generate & Precision-Trim Clips for Each Beat
        trimmed_clips = []
        for idx, beat in enumerate(beats):
            beat_num = idx + 1
            prompt = beat.get("video_prompt", "")
            target_duration = float(beat.get("duration", 3.0))

            update_progress(f"🎥 Generating visual beat {beat_num}/{len(beats)} ({target_duration:.1f}s)...")

            raw_clip_path = None
            max_retries = 3
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw_clip_path = self.video_provider.generate_video_clip(
                        prompt=prompt,
                        duration=5, # Request 5s base clip from model
                        width=1080,
                        height=1920
                    )
                    break
                except Exception as e:
                    last_err = e
                    print(f"[!] Beat {beat_num} attempt {attempt} failed: {e}")

            if not raw_clip_path:
                raise RuntimeError(f"Visual beat {beat_num} failed after {max_retries} attempts: {last_err}")

            # Precision trim clip to beat duration using FFmpeg
            trimmed_clip_path = os.path.join(clips_dir, f"beat_{beat_num:03d}.mp4")
            self._trim_clip_to_duration(raw_clip_path, trimmed_clip_path, target_duration)
            trimmed_clips.append(trimmed_clip_path)

            update_progress(f"✅ Visual beat {beat_num} complete ({target_duration:.1f}s)")

            manifest["beats"].append({
                "beat": beat_num,
                "duration": target_duration,
                "spoken_narration": beat.get("spoken_narration", ""),
                "prompt": prompt,
                "file": trimmed_clip_path
            })

        with open(os.path.join(job_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # 4. Concatenate Trimmed Beat Clips into Master Video Stream
        update_progress("🎞️ Concatenating story beats into master video...")
        concat_txt = os.path.join(job_dir, "concat.txt")
        with open(concat_txt, "w") as f:
            for clip in trimmed_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        concat_video = os.path.join(job_dir, "combined.mp4")
        ffmpeg_concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
            concat_video
        ]
        res_concat = subprocess.run(ffmpeg_concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res_concat.returncode != 0:
            raise RuntimeError(f"FFmpeg beat concatenation failed: {res_concat.stderr.decode()}")

        # 5. Merge Master Video + Master Narration Audio
        video_with_audio = os.path.join(job_dir, "video_audio.mp4")
        ffmpeg_audio_cmd = [
            "ffmpeg", "-y",
            "-i", concat_video,
            "-i", audio_file,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            video_with_audio
        ]
        res_audio = subprocess.run(ffmpeg_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res_audio.returncode != 0:
            raise RuntimeError(f"FFmpeg audio overlay failed: {res_audio.stderr.decode()}")

        # 6. Generate Synchronized Subtitles & Render Final Short
        update_progress("📝 Creating synchronized captions...")
        ass_file = os.path.join(job_dir, "captions.ass")
        self.caption_renderer.create_ass_subtitles(story_text, duration_sec=audio_duration, output_ass=ass_file)

        update_progress("🎬 Rendering final 1080x1920 Short...")
        final_short = os.path.abspath(os.path.join(job_dir, "final_short.mp4"))
        self.caption_renderer.burn_captions(video_with_audio, ass_file, final_short)

        # 7. Final Stream & Motion Validation
        self._validate_final_video(final_short, expected_audio_duration=audio_duration)

        # Output Path & URL Logging
        file_size_bytes = os.path.getsize(final_short)
        public_vps_ip = os.getenv("VPS_IP", "http://152.53.124.111:5454").rstrip("/")
        public_url = f"{public_vps_ip}/output/jobs/{job_id}/final_short.mp4"

        print("\n==================================================")
        print(f"[OUTPUT] Absolute path: {final_short}")
        print(f"[OUTPUT] Exists: {os.path.exists(final_short)}")
        print(f"[OUTPUT] Size: {file_size_bytes} bytes ({file_size_bytes / (1024*1024):.2f} MB)")
        print(f"[OUTPUT] URL: {public_url}")
        print("==================================================")

        # HTTP URL Validation (Self-test via curl)
        self._test_url_accessibility(public_url, job_id, final_short)

        update_progress("✅ Video complete!")
        return final_short

    def _trim_clip_to_duration(self, input_mp4: str, output_mp4: str, target_duration: float):
        """Precision trims/scales clip to match story beat duration exactly."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", "0", "-i", input_mp4,
            "-t", str(target_duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
            output_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def _get_audio_duration(self, audio_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "csv=p=0",
            audio_path
        ]
        res = subprocess.check_output(cmd).decode().strip()
        return float(res) if res else 0.0

    def _validate_final_video(self, video_path: str, expected_audio_duration: float):
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1024:
            raise ValueError(f"Final MP4 file missing or empty: {video_path}")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration",
            "-of", "json",
            video_path
        ]
        res = subprocess.check_output(cmd).decode()
        data = json.loads(res)
        streams = data.get("streams", [])

        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not v_stream or not a_stream:
            raise ValueError("Validation failed: Missing video or audio stream.")

        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))
        duration = float(v_stream.get("duration", 0) or 0)

        if width != 1080 or height != 1920:
            raise ValueError(f"Validation failed: Resolution is {width}x{height}, expected 1080x1920.")
        if duration <= 0:
            raise ValueError("Validation failed: Video duration is 0.")

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
            raise ValueError("Validation failed: Fewer than 5 frames.")

        diff = cv2.absdiff(frames[0], frames[-1])
        mean_diff = float(np.mean(diff))
        if mean_diff < 1.0:
            raise ValueError(f"Validation failed: Motion score {mean_diff:.2f} indicates static video.")

        print(f"[VALIDATION] Passed: {width}x{height} @ {duration:.2f}s, Audio present, Motion score={mean_diff:.2f}")

    def _test_url_accessibility(self, public_url: str, job_id: str, local_file_path: str):
        local_url = f"http://127.0.0.1:5454/output/jobs/{job_id}/final_short.mp4"
        print(f"[URL TEST] Testing local HTTP endpoint: {local_url}")

        try:
            res_local = requests.head(local_url, timeout=5)
            print(f"[URL TEST] Local HTTP Status: {res_local.status_code} (Content-Length: {res_local.headers.get('Content-Length')})")
            if res_local.status_code != 200:
                print(f"[URL TEST] WARNING: Local HTTP endpoint returned {res_local.status_code}")
        except Exception as e:
            print(f"[URL TEST] Local HTTP check error: {e}")

        try:
            res_pub = requests.head(public_url, timeout=5)
            print(f"[URL TEST] Public HTTP Status: {res_pub.status_code} (Content-Length: {res_pub.headers.get('Content-Length')})")
        except Exception as e:
            print(f"[URL TEST] Public HTTP check notice: {e}")
