import os
import logging
import subprocess

logger = logging.getLogger("RENDERER")


def render_final_video(video_clips: list, audio_path: str, subtitle_path: str, output_mp4: str = "short_output.mp4") -> str:
    """Stitches pre-rendered vertical 1080x1920 AI motion clips with voiceover and ASS subtitles."""
    valid_clips = [c for c in video_clips if c and os.path.exists(c)]

    if not valid_clips:
        logger.error("[RENDERER] [VIDEO_ENCODING_ERROR] No valid video clips available for composition.")
        raise FileNotFoundError("[VIDEO_ENCODING_ERROR] No valid video clips found to render!")

    with open("clips.txt", "w") as f:
        for clip in valid_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "clips.txt",
        "-i", audio_path,
        "-vf", f"subtitles={subtitle_path}",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192000",
        "-shortest",
        output_mp4
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"[RENDERER] [VIDEO_ENCODING_ERROR] FFmpeg composition failed:\n{res.stderr}")
        raise RuntimeError(f"[VIDEO_ENCODING_ERROR] FFmpeg composition Error: {res.stderr[-300:]}")

    if os.path.exists("clips.txt"):
        os.remove("clips.txt")

    logger.info(f"[RENDERER] Composition complete: {output_mp4}")
    return output_mp4
