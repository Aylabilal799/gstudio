import os
import logging
import discord
import asyncio
from dotenv import load_dotenv
from tts import generate_audio
from videos import generate_genuine_ai_video
from subtitles import create_glowing_subtitles
from renderer import render_final_video
from server import run_server_in_background

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
VPS_IP = os.getenv("VPS_IP", "http://152.53.124.111:5454")

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

# Mutex lock to process one generation job at a time
generation_lock = asyncio.Lock()

run_server_in_background()


@bot.event
async def on_ready():
    logger.info(f'✅ Bot active and logged in as {bot.user}')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!genshort'):
        script_text = message.content.replace('!genshort', '').strip()

        if not script_text:
            await message.channel.send("❌ Please provide a script! Example: `!genshort A detective looking for clues...`")
            return

        if generation_lock.locked():
            await message.channel.send("⏳ Another video is currently generating. Your request is queued...")

        async with generation_lock:
            status_msg = await message.channel.send("🎬 **Processing YouTube Short...** (Step 1/4: Kokoro Voiceover)")

            try:
                # 1. Voiceover
                audio_file = "voice.wav"
                await asyncio.to_thread(generate_audio, script_text, audio_file)

                # 2. Genuine AI Video Scenes
                await status_msg.edit(content="🎥 **Generating Genuine AI Video Motion...** (Step 2/4)")
                v1 = await asyncio.to_thread(
                    generate_genuine_ai_video,
                    "A character investigating confidential files in a dark room",
                    "clip1.mp4"
                )
                v2 = await asyncio.to_thread(
                    generate_genuine_ai_video,
                    "A character looking around shocked with eyes moving",
                    "clip2.mp4"
                )
                v3 = await asyncio.to_thread(
                    generate_genuine_ai_video,
                    "A character walking through a foggy dark city street at night",
                    "clip3.mp4"
                )
                clips = [v1, v2, v3]

                # 3. Subtitles
                await status_msg.edit(content="📝 **Creating Green-Glow Subtitles...** (Step 3/4)")
                sub_file = await asyncio.to_thread(create_glowing_subtitles, audio_file, "subs.ass")

                # 4. Final Composition
                await status_msg.edit(content="⚙️ **Stitching Final 1080x1920 Short...** (Step 4/4)")
                output_video = "short_output.mp4"
                await asyncio.to_thread(render_final_video, clips, audio_file, sub_file, output_video)

                download_url = f"{VPS_IP}/{output_video}"
                await status_msg.edit(content=f"✅ **Your Genuine AI Short is Ready!**\n📥 **Download Video:** {download_url}")

            except Exception as e:
                logger.error(f"[MAIN] Generation failed: {e}")
                await status_msg.edit(content=f"❌ **Error generating video:** {str(e)}")

bot.run(TOKEN)
