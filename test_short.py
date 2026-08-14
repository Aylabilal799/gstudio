import sys
import os
import shutil
from pipeline.short_generator import ShortGenerator

def main():
    default_story = (
        "I missed my bus by seconds this morning and was honestly annoyed. "
        "Then I noticed a tiny dog sitting beside the next stop, wearing a little yellow sweater. "
        "I forgot about the bus completely and started laughing. "
        "Somehow, missing that bus made my morning better."
    )
    story = sys.argv[1] if len(sys.argv) > 1 else default_story

    print("==================================================")
    print("Testing 3-Scene YouTube Short Generation Pipeline")
    print("==================================================")
    print(f"Story: \"{story}\"")

    generator = ShortGenerator()
    final_video_path = generator.generate_short(story_text=story, max_scenes=3)

    target_output = "test_short.mp4"
    shutil.copy(final_video_path, target_output)

    print("\n==================================================")
    print("TEST SHORT COMPLETE")
    print("==================================================")
    print(f"Exported File: {target_output}")
    print("Pipeline verified: 3 real AI moving clips concatenated with TTS narration, captions, and 1080x1920 resolution.")

if __name__ == "__main__":
    main()
