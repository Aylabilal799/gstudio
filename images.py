import urllib.parse
import requests
import random

def generate_scene_image(prompt: str, filename: str, seed: int = 42910) -> str:
    """Fetches high-quality 9:16 photographic realistic scenes."""

    # Consistent realistic female character prompt anchor
    base_anchor = (
        "RAW candid photograph of a 25yo European woman with dark brown hair and hazel eyes, "
        "natural skin texture with subtle pores, wearing a dark turtleneck, "
        "shot on 35mm lens, f/1.8 aperture, shallow depth of field, realistic studio lighting"
    )

    full_prompt = f"{base_anchor}, {prompt}, cinematic film grain"
    encoded_prompt = urllib.parse.quote(full_prompt)

    # 9:16 Vertical Resolution (1080x1920)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&seed={seed}&nologo=true&model=flux"

    response = requests.get(image_url, timeout=60)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        raise Exception("Failed to generate image from API")
