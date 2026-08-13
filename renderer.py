import os
import subprocess

def render_final_video(video_clips: list, audio_path: str, subtitle_path: str, output_mp4: str = "short_output.mp4"):
    """Stitches normalized MP4 video clips, adds Kokoro audio, and burns ASS subtitles."""

    valid_clips = [c for c in video_clips if c and os.path.exists(c)]

    if not valid_clips:
        raise FileNotFoundError("No valid video clips found to render!")

    with open("clips.txt", "w") as f:
        for clip in valid_clips:
            f.write(f"file '{clip}'\n")

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
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192000",
        "-shortest",
        output_mp4
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ FFmpeg Failure Log:\n", result.stderr)
        raise RuntimeError(f"FFmpeg Error: {result.stderr[-300:]}")

    return output_mp4
