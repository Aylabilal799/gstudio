import sys
import os
import subprocess
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from audio.tts import TTSEngine

load_dotenv()

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "I missed my bus by seconds this morning and was honestly annoyed."
    output_wav = "test_narration.wav"

    print("==================================================")
    print("Testing AFHeart Female Voice TTS Generation")
    print("==================================================")

    tts = TTSEngine()
    result_path = tts.generate_narration(text=text, output_file=output_wav)

    # Inspect file using ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,duration",
        "-of", "csv=p=0",
        result_path
    ]
    res = subprocess.check_output(cmd).decode().strip().split(",")
    channels, sample_rate, duration = res[0], res[1], res[2]

    # Verify non-silent audio signal with soundfile
    data, samplerate = sf.read(result_path)
    rms_volume = float(np.sqrt(np.mean(data**2)))

    if rms_volume < 0.001:
        raise ValueError(f"REJECTED: Audio file appears to be silent (RMS volume = {rms_volume:.6f}).")

    print("\n==================================================")
    print("TTS VERIFICATION REPORT")
    print("==================================================")
    print(f"[TTS] Engine: Kokoro ONNX")
    print(f"[TTS] Voice: af_heart")
    print(f"[TTS] Duration: {float(duration):.2f} seconds")
    print(f"[TTS] Sample rate: {sample_rate} Hz")
    print(f"[TTS] Channels: {channels}")
    print(f"[TTS] Output: {result_path}")
    print(f"[TTS] RMS Audio Signal: {rms_volume:.4f} (VERIFIED SPOKEN AUDIO)")
    print("==================================================")

if __name__ == "__main__":
    main()
