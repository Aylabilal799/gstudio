from pipeline.short_generator import ShortGenerator

def test_short_pipeline():
    story = (
        "I missed my bus by seconds this morning and was honestly annoyed. "
        "Then I noticed a tiny dog sitting beside the next stop, wearing a little yellow sweater. "
        "I forgot about the bus completely and started laughing. "
        "Somehow, missing that bus made my morning better."
    )
    print("[*] Running Short Pipeline Test (2-3 Scenes)...")

    generator = ShortGenerator()
    final_mp4 = generator.generate_short(story_text=story, max_scenes=3)

    print(f"\n[+] TEST SHORT PIPELINE COMPLETE: {final_mp4}")

if __name__ == "__main__":
    test_short_pipeline()
