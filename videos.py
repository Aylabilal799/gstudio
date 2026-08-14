import os
import sys
import gc
import logging
import subprocess
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VIDEO")

# Configurable Parameters
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "THUDM/CogVideoX-2B")
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "720"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))
VIDEO_FRAMES = int(os.getenv("VIDEO_FRAMES", "49"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "8"))
VIDEO_STEPS = int(os.getenv("VIDEO_STEPS", "20"))
VIDEO_DEVICE_STR = os.getenv("VIDEO_DEVICE", "cpu").lower()
VIDEO_OUTPUT_WIDTH = int(os.getenv("VIDEO_OUTPUT_WIDTH", "1080"))
VIDEO_OUTPUT_HEIGHT = int(os.getenv("VIDEO_OUTPUT_HEIGHT", "1920"))
VIDEO_REFRAME_MODE = os.getenv("VIDEO_REFRAME_MODE", "crop_center")

# Centralized Device Configuration
DEVICE = torch.device(VIDEO_DEVICE_STR)

# Default to bfloat16 for 50% memory savings (~9.5GB peak virtual memory)
VIDEO_DTYPE_STR = os.getenv("VIDEO_DTYPE", "bfloat16" if DEVICE.type == "cpu" else "bfloat16").lower()

if VIDEO_DTYPE_STR == "bfloat16":
    TORCH_DTYPE = torch.bfloat16
elif VIDEO_DTYPE_STR == "float16":
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32

MOTION_PROMPT_LAYER = (
    "Continuous natural temporal animation. The subject moves organically, "
    "body parts and facial features animate smoothly over time, clothing and hair react "
    "naturally to motion, background elements have subtle continuous movement. "
    "Realistic video motion, no static freeze, no still image."
)

_pipeline_cache = {}


def check_memory_status(tag: str = "General"):
    """Logs system RAM and Swap availability."""
    try:
        import psutil
        proc = psutil.Process()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        rss_gb = proc.memory_info().rss / (1024**3)
        ram_gb = mem.available / (1024**3)
        swap_free_gb = swap.free / (1024**3)
        swap_used_gb = swap.used / (1024**3)
        logger.info(
            f"[VIDEO] [{tag}] RSS RAM: {rss_gb:.2f} GB | "
            f"RAM Free: {ram_gb:.2f} GB | Swap Used: {swap_used_gb:.2f} GB | Swap Free: {swap_free_gb:.2f} GB"
        )
    except Exception:
        pass


def get_cogvideox_pipeline(image_to_video=False):
    """Loads and initializes CogVideoX pipeline strictly from local HuggingFace cache."""
    key = "i2v" if image_to_video else "t2v"
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    check_memory_status("Before Pipeline Init")
    logger.info(f"[VIDEO] Device: {DEVICE.type}")
    logger.info(f"[VIDEO] Model: {VIDEO_MODEL}")
    logger.info(f"[VIDEO] Pipeline Type: {'Image-to-Video' if image_to_video else 'Text-to-Video'}")
    logger.info(f"[VIDEO] Torch Dtype: {TORCH_DTYPE}")

    try:
        from diffusers import CogVideoXImageToVideoPipeline, CogVideoXPipeline

        kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "low_cpu_mem_usage": True,
            "local_files_only": True
        }

        logger.info("[VIDEO] Loading pipeline components strictly from LOCAL CACHE (local_files_only=True)...")

        # EXACTLY ONE PIPELINE INSTANCE - NO RETRY FALLBACK
        if image_to_video:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(VIDEO_MODEL, **kwargs)
        else:
            pipe = CogVideoXPipeline.from_pretrained(VIDEO_MODEL, **kwargs)

        logger.info("[VIDEO] Model source: LOCAL CACHE")
        check_memory_status("After Weights Loaded")

        pipe = pipe.to(DEVICE)
        check_memory_status("After Pipe to CPU")

        # Enable VAE Slicing & Tiling to minimize RAM during VAE decoding
        try:
            if hasattr(pipe, "vae"):
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                logger.info("[VIDEO] VAE slicing & tiling: enabled")
        except Exception:
            pass

        _pipeline_cache[key] = pipe
        return pipe

    except Exception as e:
        logger.exception(f"[VIDEO] [MODEL_LOAD_ERROR] CogVideoX initialization failed from local cache: {e}")
        raise RuntimeError(f"[MODEL_LOAD_ERROR] CogVideoX initialization failed from local cache ({e.__class__.__name__}): {e}") from e


def reframe_to_vertical(input_video_path: str, output_video_path: str) -> str:
    """Intelligently converts landscape 720x480 clip to vertical 1080x1920 (9:16) at 30 FPS."""
    logger.info(f"[VIDEO] Reframing source to: {VIDEO_OUTPUT_WIDTH}x{VIDEO_OUTPUT_HEIGHT} ({VIDEO_REFRAME_MODE})")

    if VIDEO_REFRAME_MODE == "blur_padding":
        vf_filter = (
            f"split[main][bg];"
            f"[bg]scale={VIDEO_OUTPUT_WIDTH}:{VIDEO_OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_OUTPUT_WIDTH}:{VIDEO_OUTPUT_HEIGHT},boxblur=25:5[bgout];"
            f"[main]scale={VIDEO_OUTPUT_WIDTH}:{VIDEO_OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
            f"[bgout][fg]overlay=(W-w)/2:(H-h)/2,fps=30"
        )
    else:
        # Default: crop_center
        vf_filter = (
            f"scale=-1:{VIDEO_OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_OUTPUT_WIDTH}:{VIDEO_OUTPUT_HEIGHT},fps=30"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-an",
        output_video_path
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"[VIDEO] [VIDEO_ENCODING_ERROR] FFmpeg reframing failed:\n{res.stderr}")
        raise RuntimeError(f"[VIDEO_ENCODING_ERROR] Reframing video failed: {res.stderr[-300:]}")

    return output_video_path


def generate_genuine_ai_video(
    prompt: str,
    output_path: str,
    image_path: str = None
) -> str:
    """Generates genuine temporal AI video frames using official Diffusers CogVideoX pipeline API."""
    full_prompt = f"{prompt}. {MOTION_PROMPT_LAYER}"

    logger.info(f"[VIDEO] Device: {DEVICE.type}")
    logger.info(f"[VIDEO] Model: {VIDEO_MODEL}")
    logger.info(f"[VIDEO] Model source: LOCAL CACHE")
    logger.info(f"[VIDEO] CPU offload: disabled")
    logger.info(f"[VIDEO] Starting genuine AI video generation")
    logger.info(f"[VIDEO] Source resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

    temp_raw_video = f"raw_{os.path.basename(output_path)}"

    try:
        from diffusers.utils import export_to_video, load_image

        use_i2v = bool(image_path and os.path.exists(image_path))
        pipe = get_cogvideox_pipeline(image_to_video=use_i2v)

        logger.info(f"[VIDEO] Executing CogVideoX generation pipeline ({VIDEO_STEPS} steps)...")

        generator = torch.Generator(device="cpu").manual_seed(42)

        if use_i2v:
            image_input = load_image(image_path)
            output = pipe(
                image=image_input,
                prompt=full_prompt,
                height=VIDEO_HEIGHT,
                width=VIDEO_WIDTH,
                num_frames=VIDEO_FRAMES,
                num_inference_steps=VIDEO_STEPS,
                generator=generator,
                output_type="pil"
            )
        else:
            output = pipe(
                prompt=full_prompt,
                height=VIDEO_HEIGHT,
                width=VIDEO_WIDTH,
                num_frames=VIDEO_FRAMES,
                num_inference_steps=VIDEO_STEPS,
                generator=generator,
                output_type="pil"
            )

        pil_frames = output.frames[0]
        num_generated_frames = len(pil_frames)

        logger.info(f"[VIDEO] Diffusion complete")
        logger.info(f"[VIDEO] VAE decode complete")

        # Inspect first PIL frame
        first_frame_np = np.array(pil_frames[0])
        f_min = int(first_frame_np.min())
        f_max = int(first_frame_np.max())
        f_mean = float(first_frame_np.mean())
        f_std = float(first_frame_np.std())

        logger.info(f"[VIDEO] Pipeline output type: {type(pil_frames[0])}")
        logger.info(f"[VIDEO] Pipeline frame shape: {first_frame_np.shape}")
        logger.info(f"[VIDEO] Pipeline frame dtype: {first_frame_np.dtype}")
        logger.info(f"[VIDEO] Pipeline frame min: {f_min}")
        logger.info(f"[VIDEO] Pipeline frame max: {f_max}")
        logger.info(f"[VIDEO] Pipeline frame mean: {f_mean:.2f}")
        logger.info(f"[VIDEO] Pipeline frame std: {f_std:.2f}")

        # Save Raw Debug Frames directly from PIL output BEFORE export/FFmpeg
        debug_path_0 = "debug_pipeline_frame.png"
        pil_frames[0].save(debug_path_0)
        logger.info(f"[VIDEO] First decoded frame saved: {debug_path_0}")

        f10_idx = min(10, num_generated_frames - 1)
        f20_idx = min(20, num_generated_frames - 1)
        pil_frames[f10_idx].save(f"debug_frame_{f10_idx:03d}.png")
        pil_frames[f20_idx].save(f"debug_frame_{f20_idx:03d}.png")

        # Strict non-black verification
        if f_max == 0:
            logger.error("[VIDEO] CRITICAL: Decoded frames are completely black (max=0)!")
            raise RuntimeError("[INFERENCE_ERROR] Generated video frames are completely black. Numerical instability or dtype underflow.")

        # Export non-black PIL frames to raw landscape MP4 video
        export_to_video(pil_frames, temp_raw_video, fps=VIDEO_FPS)

        # Convert landscape 720x480 to vertical 1080x1920 (9:16)
        reframe_to_vertical(temp_raw_video, output_path)

        if os.path.exists(temp_raw_video):
            os.remove(temp_raw_video)

        gc.collect()

        logger.info(f"[VIDEO] Model: {VIDEO_MODEL}")
        logger.info(f"[VIDEO] Device: {DEVICE.type.upper()}")
        logger.info(f"[VIDEO] Model source: LOCAL CACHE")
        logger.info(f"[VIDEO] Diffusion steps: {VIDEO_STEPS}")
        logger.info(f"[VIDEO] Generated frames: {num_generated_frames}")
        logger.info(f"[VIDEO] Source resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
        logger.info(f"[VIDEO] Final resolution: {VIDEO_OUTPUT_WIDTH}x{VIDEO_OUTPUT_HEIGHT}")
        logger.info(f"[VIDEO] Genuine AI video: YES")
        logger.info(f"[VIDEO] Output: {output_path}")

        return output_path

    except Exception as e:
        logger.exception(f"[VIDEO] [INFERENCE_ERROR] Genuine AI video generation failed: {e}")
        if os.path.exists(temp_raw_video):
            os.remove(temp_raw_video)
        raise RuntimeError(f"[INFERENCE_ERROR] Genuine AI video generation failed ({e.__class__.__name__}): {e}") from e
