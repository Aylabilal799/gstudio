import soundfile as sf
from kokoro import KPipeline

# Initialize official Kokoro pipeline ('a' = American English)
# This auto-fetches official weights natively from Hugging Face
pipeline = KPipeline(lang_code='a')

def generate_audio(text: str, output_path: str = "voice.wav") -> str:
    """Generates speech locally using official hexgrad/kokoro with 'af_heart' voice."""
    print("🎙️ Synthesizing voice with 'af_heart' via official Kokoro...")

    generator = pipeline(
        text,
        voice='af_heart',
        speed=1.0,
        split_pattern=r'\n+'
    )

    # Collect generated audio chunks
    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.extend(audio)

    sf.write(output_path, audio_chunks, 24000)
    return output_path

if __name__ == "__main__":
    generate_audio("Testing official Kokoro local generation.", "test.wav")
