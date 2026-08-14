import os
from audio.afheart_tts import AFHeartTTS

class TTSEngine:
    def __init__(self, voice: str = None):
        self.voice = voice or os.getenv("TTS_VOICE", "af_heart")
        self.engine = AFHeartTTS(voice=self.voice)

    def generate_narration(self, text: str, output_file: str = "output/narration.wav") -> str:
        return self.engine.generate_narration(text=text, output_file=output_file)
