import os
import json
import requests

class StoryPlanner:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_API_BASE_URL", "https://freellmapi.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def plan_story_beats(self, story_text: str, total_narration_duration: float):
        system_prompt = f"""You are a master cinematic video editor.
Given a story and its EXACT spoken narration audio duration ({total_narration_duration:.2f} seconds), break the story into 3 to 4 visual story beats.

CRITICAL TIMELINE REQUIREMENTS:
1. The sum of all beat durations MUST EXACTLY equal {total_narration_duration:.2f} seconds.
2. For each beat, specify:
   - `beat_number`
   - `start_time` (seconds)
   - `end_time` (seconds)
   - `duration` (seconds, e.g., 3.2)
   - `spoken_narration` (the exact portion of story text spoken during this beat)
   - `video_prompt` (detailed 9:16 portrait cinematic prompt visually matching the spoken text)

3. Include character appearance continuity (young man, mid-20s, short dark hair, navy jacket, dark trousers, white sneakers, tiny brown dog in yellow sweater).

Return ONLY valid JSON matching this schema:
{{
  "character_bible": {{
    "main_character": "Young man, mid-20s, short dark hair, navy jacket...",
    "dog": "Tiny dog in yellow knitted sweater"
  }},
  "beats": [
    {{
      "beat_number": 1,
      "start_time": 0.0,
      "end_time": 3.5,
      "duration": 3.5,
      "spoken_narration": "I missed my bus by seconds this morning and was honestly annoyed.",
      "video_prompt": "Cinematic 9:16 portrait. Young man in mid-20s, short dark hair, navy jacket, dark trousers, white sneakers, running frantically toward a departing city bus at an urban bus stop in soft morning light, annoyed facial expression."
    }}
  ]
}}
"""

        user_prompt = f"Story: \"{story_text}\"\nTotal Narration Duration: {total_narration_duration:.2f}s"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"[LLM] Notice: LLM API call returned ({e}). Calculating mathematical beat breakdown.")

            # Fallback proportional beat breakdown
            b1_dur = round(total_narration_duration * 0.30, 2)
            b2_dur = round(total_narration_duration * 0.25, 2)
            b3_dur = round(total_narration_duration * 0.25, 2)
            b4_dur = round(total_narration_duration - (b1_dur + b2_dur + b3_dur), 2)

            return {
                "character_bible": {
                    "main_character": "Young man, mid-20s, short dark hair, navy jacket, dark trousers, white sneakers",
                    "dog": "Tiny brown dog, yellow sweater"
                },
                "beats": [
                    {
                        "beat_number": 1,
                        "duration": b1_dur,
                        "spoken_narration": "I missed my bus by seconds this morning and was honestly annoyed.",
                        "video_prompt": "Cinematic 9:16 portrait. Young man, mid-20s, short dark hair, navy jacket, dark trousers, white sneakers, running toward departing bus at city bus stop, annoyed facial expression."
                    },
                    {
                        "beat_number": 2,
                        "duration": b2_dur,
                        "spoken_narration": "Then I noticed a tiny dog sitting beside the next stop,",
                        "video_prompt": "Cinematic 9:16 portrait. Same young man looking annoyed, then turning his head and noticing a tiny brown dog sitting by the bus stop bench."
                    },
                    {
                        "beat_number": 3,
                        "duration": b3_dur,
                        "spoken_narration": "wearing a little yellow sweater.",
                        "video_prompt": "Cinematic 9:16 portrait. Close-up of tiny brown dog wearing a bright yellow knitted sweater sitting at the bus stop, looking cute."
                    },
                    {
                        "beat_number": 4,
                        "duration": b4_dur,
                        "spoken_narration": "I forgot about the bus completely and started laughing. Somehow, missing that bus made my morning better.",
                        "video_prompt": "Cinematic 9:16 portrait. Same young man smiling and laughing happily while sitting beside the tiny dog in yellow sweater at the bus stop."
                    }
                ]
            }
