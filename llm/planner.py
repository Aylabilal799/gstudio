import os
import json
import requests

class StoryPlanner:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_API_BASE_URL", "https://freellmapi.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def plan_story(self, story_text: str, default_clip_duration: int = 5, max_scenes: int = 5):
        system_prompt = """You are an expert AI video scene director.
Your job is to analyze a user story and produce a CHARACTER BIBLE and a structured SCENE BREAKDOWN for text-to-video generation.

CRITICAL INSTRUCTIONS:
1. Create a detailed Character Bible describing character physical traits, hair, age, clothing, and footwear.
2. Break the story into short cinematic video scenes (each 3 to 5 seconds long).
3. Every scene's prompt MUST contain:
   - Character Bible details (exact clothing, hair, age, appearance for visual continuity)
   - Specific body actions, facial expressions, camera movement, environment, lighting, and time of day
   - Natural human movement and physically plausible action
4. Do NOT make generic prompts like "man walks". Use detailed cinematic descriptions.
5. Return ONLY a valid JSON object.
"""

        user_prompt = f"""
Story: "{story_text}"

Max Scenes Allowed: {max_scenes}
Clip Duration: {default_clip_duration} seconds

Return JSON in this format:
{{
  "character_bible": {{
    "main_character": "detailed physical description...",
    "key_elements": "description of key items, animals, or environments..."
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "duration": {default_clip_duration},
      "video_prompt": "Cinematic 9:16 portrait prompt..."
    }}
  ]
}}
"""

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

        res = requests.post(url, headers=headers, json=payload, timeout=40)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return json.loads(content)
