import whisper

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs - int(secs)) * 100)
    return f"{hours:01d}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"

def create_glowing_subtitles(audio_path: str, output_ass: str = "subs.ass"):
    """Uses Whisper to extract timestamps and create CapCut-style Green Glow ASS subtitles."""
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, word_timestamps=True)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCutGlow,DejaVu Sans,75,&H00FFFFFF,&H0000FF00,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,100,100,850,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for segment in result['segments']:
        if 'words' in segment:
            for word_info in segment['words']:
                word = word_info['word'].strip().upper()
                start = format_time(word_info['start'])
                end = format_time(word_info['end'])

                # Dynamic Green Glow Effect
                line = f"Dialogue: 0,{start},{end},CapCutGlow,,0,0,0,,{{\\1c&H00FF00&\\blur5}}{word}"
                events.append(line)

    with open(output_ass, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

    return output_ass
