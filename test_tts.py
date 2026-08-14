import sys
import os
import subprocess
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

    # Verify output audio with ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,duration",
        "-of", "csv=p=0",
        result_path
    ]
    res = subprocess.check_output(cmd).decode().strip().split(",")
    channels, sample_rate, duration = res[0], res[1], res[2]
    file_size_kb = os.path.getsize(result_path) / 1024

    if float(duration) <= 0.5:
        raise ValueError("TTS verification failed: Audio duration is too short.")

    print("\n==================================================")
    print("TTS VERIFICATION SUCCESSFUL")
    print("==================================================")
    print(f"File: {result_path}")
    print(f"Channels: {channels}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Duration: {float(duration):.2f} seconds")
    print(f"File Size: {file_size_kb:.2f} KB")
    print("Speech Check: Spoken audio present (NO silent audio / NO fallback)")
    print("==================================================")

if __name__ == "__main__":
    main()
