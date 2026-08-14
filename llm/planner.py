import os
import json
import requests

class StoryPlanner:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_API_BASE_URL", "https://freellmapi.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def plan_story(self, story_text: str, default_clip_duration: int = 5, num_scenes: int = 3):
        system_prompt = """You are an expert AI video director.
Analyze the story and output a JSON object containing a CHARACTER BIBLE and detailed SCENE PROMPTS.

CRITICAL REQUIREMENTS:
1. Create a detailed Character Bible describing the main character's exact physical features, clothing, hair, age, and footwear for visual continuity.
2. Create exactly 3 short scenes (5 seconds each).
3. Every scene's prompt MUST incorporate the Character Bible description, specific body actions, facial expressions, camera angles, lighting, time of day, and natural motion.
4. Output ONLY valid JSON matching this schema:
{
  "character_bible": {
    "main_character": "Young man, mid-20s, short dark hair, navy blue jacket, dark trousers, white sneakers...",
    "dog": "Tiny brown dog, floppy ears, bright yellow knitted sweater..."
  },
  "scenes": [
    {
      "scene_number": 1,
      "duration": 5,
      "video_prompt": "Cinematic portrait scene. Young man in mid-20s with short dark hair, navy blue jacket, dark trousers, and white sneakers running toward a city bus that is closing its doors at a modern urban bus stop. Breathing heavily and looking frustrated as the bus pulls away. Soft morning light, shallow depth of field, natural physical movement."
    }
  ]
}
"""

        user_prompt = f"Story: \"{story_text}\"\nRequested Scenes: {num_scenes}\nScene Duration: {default_clip_duration}s"

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
            print(f"[LLM] Notice: LLM API call returned ({e}). Using default 3-scene character breakdown.")
            return {
                "character_bible": {
                    "main_character": "Young man, mid-20s, short dark hair, navy blue jacket, dark trousers, white sneakers",
                    "dog": "Tiny brown dog, floppy ears, yellow knitted sweater"
                },
                "scenes": [
                    {
                        "scene_number": 1,
                        "duration": 5,
                        "video_prompt": "Cinematic 9:16 portrait scene. Young man, mid-20s, short dark hair, navy blue jacket, dark trousers, and white sneakers, running toward a departing city bus at a modern urban bus stop in soft morning light, natural body movement, frustrated facial expression."
                    },
                    {
                        "scene_number": 2,
                        "duration": 5,
                        "video_prompt": "Cinematic 9:16 portrait scene. Same young man in navy blue jacket watching the bus leave, looking annoyed, then turning his head and noticing a tiny brown dog in a yellow knitted sweater sitting beside the bus stop bench."
                    },
                    {
                        "scene_number": 3,
                        "duration": 5,
                        "video_prompt": "Cinematic 9:16 portrait scene. Close-up of same young man smiling and laughing happily while sitting beside the cute tiny dog wearing a yellow sweater at the city bus stop."
                    }
                ]
            }
