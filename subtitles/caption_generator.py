import os
import shutil
import subprocess

class CaptionRenderer:
    @staticmethod
    def create_ass_subtitles(text: str, duration_sec: float, output_ass: str) -> str:
        os.makedirs(os.path.dirname(output_ass), exist_ok=True)
        words = text.split()
        chunk_size = 5
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        chunk_duration = duration_sec / max(len(chunks), 1)

        ass_header = """[Script Info]
Title: GStudio Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,2,60,60,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        for idx, chunk in enumerate(chunks):
            start_time = idx * chunk_duration
            end_time = (idx + 1) * chunk_duration

            def format_time(t):
                hrs = int(t // 3600)
                mins = int((t % 3600) // 60)
                secs = int(t % 60)
                ms = int((t % 1) * 100)
                return f"{hrs}:{mins:02d}:{secs:02d}.{ms:02d}"

            start_str = format_time(start_time)
            end_str = format_time(end_time)
            events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\b1}}{chunk}{{\\b0}}")

        with open(output_ass, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(events))

        return output_ass

    @staticmethod
    def burn_captions(video_input: str, ass_subtitles: str, video_output: str) -> str:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_input,
            "-vf", f"ass='{ass_subtitles}'",
            "-c:v", "libx264", "-c:a", "copy",
            video_output
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(video_output):
            print("[CAPTIONS] Notice: Copying final video without ASS filter overlay.")
            shutil.copy(video_input, video_output)
        return video_output
