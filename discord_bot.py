import os
import time
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv

from pipeline.short_generator import ShortGenerator

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
VPS_IP = os.getenv("VPS_IP", "http://152.53.124.111:5454").rstrip("/")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Single job concurrency lock
job_lock = asyncio.Lock()
short_generator = None

class ThrottledProgressManager:
    """Coalesces rapid internal progress events to update Discord at most once every 4 seconds."""
    def __init__(self, status_msg, loop, min_interval: float = 4.0):
        self.status_msg = status_msg
        self.loop = loop
        self.min_interval = min_interval
        self.last_update_time = 0.0
        self.latest_text = None
        self.lock = asyncio.Lock()
        self._last_sent_text = None

    def update(self, new_text: str):
        self.latest_text = new_text
        asyncio.run_coroutine_threadsafe(self._schedule_flush(), self.loop)

    async def _schedule_flush(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

            if self.latest_text and self.latest_text != self._last_sent_text:
                await self._safe_edit(self.latest_text)

    async def _safe_edit(self, text: str):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.status_msg.edit(content=text)
                self.last_update_time = time.time()
                self._last_sent_text = text
                break
            except discord.HTTPException as e:
                if e.status == 429:  # Rate Limit
                    retry_after = getattr(e, "retry_after", 5.0) or 5.0
                    print(f"[DISCORD] Rate limited (429). Waiting {retry_after:.2f}s before retrying...")
                    await asyncio.sleep(retry_after)
                else:
                    print(f"[DISCORD] HTTP error editing message ({e.status}): {e}")
                    break
            except Exception as e:
                print(f"[DISCORD] Error editing status message: {e}")
                break

    async def flush_final(self, final_text: str):
        self.latest_text = final_text
        await self._safe_edit(final_text)

@bot.event
async def on_ready():
    global short_generator
    print(f"==================================================")
    print(f"GStudio Discord Bot Connected as: {bot.user}")
    print(f"==================================================")
    short_generator = ShortGenerator()

@bot.command(name="video")
async def create_video(ctx, *, story_text: str = None):
    if not story_text or not story_text.strip():
        await ctx.send("❌ Usage: `!video <story text>`\nExample: `!video I missed my bus by seconds this morning...`")
        return

    # Check job concurrency
    if job_lock.locked():
        await ctx.send("⚠️ A video job is currently in progress. Please wait until it completes.")
        return

    async with job_lock:
        status_msg = await ctx.send("🎬 Starting video generation...")
        loop = asyncio.get_running_loop()

        progress_mgr = ThrottledProgressManager(status_msg, loop, min_interval=4.0)

        def sync_progress_callback(text_msg: str):
            progress_mgr.update(text_msg)

        def run_pipeline():
            return short_generator.generate_short(
                story_text=story_text,
                max_scenes=3,  # Controlled 3-scene test limit
                progress_callback=sync_progress_callback
            )

        try:
            # Run heavy CPU pipeline in executor thread
            final_mp4 = await loop.run_in_executor(None, run_pipeline)

            file_size_mb = os.path.getsize(final_mp4) / (1024 * 1024)
            job_id = os.path.basename(os.path.dirname(final_mp4))
            public_url = f"{VPS_IP}/output/jobs/{job_id}/final_short.mp4"

            final_text = (
                f"✅ **Video complete!**\n"
                f"📌 **Job ID:** `{job_id}`\n"
                f"📐 **Resolution:** `1080x1920` (9:16 Short)\n"
                f"🎙️ **Voice:** Kokoro ONNX (`af_heart` female)\n"
                f"🔗 **Download / Watch Link:** {public_url}"
            )

            await progress_mgr.flush_final(final_text)

            # Upload MP4 file to Discord if under 25 MB
            if file_size_mb < 25.0:
                print(f"[DISCORD] Uploading final MP4 ({file_size_mb:.2f} MB) to Discord channel...")
                with open(final_mp4, "rb") as f:
                    discord_file = discord.File(f, filename="final_short.mp4")
                    await ctx.send(file=discord_file)
                print("[DISCORD] File upload successful!")
            else:
                await ctx.send(f"📦 Video file is {file_size_mb:.1f} MB. Watch via link: {public_url}")

        except Exception as e:
            print(f"[DISCORD] Pipeline Error: {e}")
            await progress_mgr.flush_final(f"❌ **Generation Failed**\nReason: `{e}`")

if __name__ == "__main__":
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN missing from environment variables.")
    bot.run(TOKEN)
