import os
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro

class AFHeartTTS:
    def __init__(self, voice: str = "af_heart", models_dir: str = "output/models"):
        self.voice = voice or os.getenv("TTS_VOICE", "af_heart")
        self.models_dir = models_dir

        self.model_path = os.path.join(self.models_dir, "kokoro-v1.0.int8.onnx")
        self.voices_path = os.path.join(self.models_dir, "voices-v1.0.bin")

        self._ensure_models_exist()

        try:
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            print(f"[TTS] Loaded AFHeart Kokoro ONNX engine on CPU with female voice '{self.voice}'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AFHeart Kokoro TTS: {e}")

    def _ensure_models_exist(self):
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 10000:
            raise RuntimeError(f"Model file missing at {self.model_path}.")
        if not os.path.exists(self.voices_path) or os.path.getsize(self.voices_path) < 10000:
            raise RuntimeError(f"Voices file missing at {self.voices_path}.")

    def generate_narration(self, text: str, output_file: str = "output/narration.wav") -> str:
        if not text or not text.strip():
            raise ValueError("TTS generation error: Input text is empty.")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"[TTS] Engine: AFHeart Kokoro")
        print(f"[TTS] Gender: female")
        print(f"[TTS] Voice: {self.voice}")
        print(f"[TTS] Generating female narration audio for text: \"{text[:60]}...\"")

        try:
            samples, sample_rate = self.kokoro.create(
                text=text.strip(),
                voice=self.voice,
                speed=1.0,
                lang="en-us"
            )

            if len(samples) == 0:
                raise RuntimeError("AFHeart TTS produced an empty audio buffer.")

            sf.write(output_file, samples, sample_rate)

            if not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
                raise RuntimeError(f"Generated audio file {output_file} is invalid or empty.")

            duration = len(samples) / float(sample_rate)
            print(f"[TTS] SUCCESS! Output: {output_file} ({duration:.2f} seconds @ {sample_rate} Hz)")
            return output_file

        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {e}")
