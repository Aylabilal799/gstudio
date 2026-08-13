# Autonomous AI YouTube Shorts Discord Studio 🎬

A 100% free, self-hosted Discord bot pipeline for generating YouTube Shorts.

## Features
- **Voiceover:** Kokoro TTS (`af_heart` voice) running 100% locally on CPU.
- **AI Video:** CogVideoX-5B AI Video generation via Hugging Face GPU Spaces.
- **Subtitles:** Whisper word-level timestamps with CapCut-style Green Glow ASS subtitles.
- **Editing:** Dynamic FFmpeg resolution normalization, motion stitching, and web server delivery.

## Setup Instructions
1. Install system dependencies: `apt install -y ffmpeg espeak-ng python3.11 python3.11-venv`
2. Create virtual environment: `python3.11 -m venv venv && source venv/bin/activate`
3. Install requirements: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in your Discord token & VPS IP.
5. Start bot: `python main.py`
