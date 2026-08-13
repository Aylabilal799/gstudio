import os
import time
import urllib.parse
import requests
import subprocess
from gradio_client import Client, handle_file

def normalize_clip(input_path: str, output_path: str, duration: float = 3.5) -> str:
    """Standardizes any generated video clip or image to 1080x1920 @ 25fps H.264."""
    is_image = input_path.lower().endswith((".jpg", ".png", ".jpeg"))

    if is_image:
        vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.002,1.2)':d={int(duration*25)}:s=1080x1920:fps=25"
    else:
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=25"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1" if is_image else "0",
        "-i", input_path,
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-an",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def generate_scene_image_with_retry(prompt: str, filename: str, seed: int = 42910) -> str:
    """Generates base scene image with 60s timeout and 3-attempt retry loop."""
    base_anchor = "photorealistic 25yo European woman with dark brown hair wearing a black turtleneck"
    full_prompt = f"{base_anchor}, {prompt}, 8k resolution, cinematic lighting"
    encoded = urllib.parse.quote(full_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&seed={seed}&nologo=true&model=flux"

    for attempt in range(3):
        try:
            print(f"🎨 Generating base scene image (Attempt {attempt+1}/3)...")
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(res.content)
                return filename
        except Exception as e:
            print(f"⚠️ Image download warning ({e}). Retrying in 2 seconds...")
            time.sleep(2)

    raise RuntimeError("Failed to download scene image after 3 attempts.")

def generate_cogvideox_clip(prompt: str, output_path: str, duration: float = 3.5, seed: int = 42910) -> str:
    """Generates scene image first, then passes image_input to CogVideoX-5B AI Video space."""

    # 1. Generate base character scene image
    temp_img = f"base_{os.path.basename(output_path)}.jpg"
    generate_scene_image_with_retry(prompt, temp_img, seed=seed)

    print(f"🎥 Passing image & prompt to CogVideoX-5B AI Video Space...")

    # 2. Call CogVideoX-5B with required image_input argument
    try:
        client = Client("THUDM/CogVideoX-5B-Space")
        result = client.predict(
            image_input=handle_file(temp_img),
            prompt=prompt,
            api_name="/generate"
        )

        generated_mp4 = result[0] if isinstance(result, (tuple, list)) else result
        if generated_mp4 and os.path.exists(generated_mp4):
            print("✅ CogVideoX-5B AI Video generated successfully!")
            res = normalize_clip(generated_mp4, output_path, duration)
            if os.path.exists(temp_img):
                os.remove(temp_img)
            return res
    except Exception as e:
        print(f"⚠️ CogVideoX-5B queue busy ({e}). Using scene clip...")

    # 3. Fallback motion clip from generated base image
    res = normalize_clip(temp_img, output_path, duration)
    if os.path.exists(temp_img):
        os.remove(temp_img)
    return res
