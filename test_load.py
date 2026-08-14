import os
import sys
import gc
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MEMORY_TEST")

def log_mem(tag: str):
    try:
        import psutil
        proc = psutil.Process()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        rss_gb = proc.memory_info().rss / (1024**3)
        avail_ram = mem.available / (1024**3)
        swap_used = swap.used / (1024**3)
        swap_free = swap.free / (1024**3)
        print(f"[MEMORY] [{tag}] RSS RAM: {rss_gb:.2f} GB | Available RAM: {avail_ram:.2f} GB | Swap Used: {swap_used:.2f} GB | Swap Free: {swap_free:.2f} GB")
    except Exception as e:
        print(f"[MEMORY] [{tag}] Memory check error: {e}")

def test_model_loading():
    print("=== Dedicated CogVideoX-2B Local Model Loading Test ===")

    device_str = os.getenv("VIDEO_DEVICE", "cpu").lower()
    dtype_str = os.getenv("VIDEO_DTYPE", "bfloat16").lower()
    model_id = os.getenv("VIDEO_MODEL", "THUDM/CogVideoX-2B")

    device = torch.device(device_str)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else (torch.float16 if dtype_str == "float16" else torch.float32)

    print(f"[MEMORY] Target Model: {model_id}")
    print(f"[MEMORY] Target Device: {device.type}")
    print(f"[MEMORY] Target Dtype: {dtype}")

    log_mem("Before pipeline initialization")

    from diffusers import CogVideoXPipeline

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True
    }

    print("[MEMORY] Loading CogVideoXPipeline strictly from LOCAL CACHE (local_files_only=True)...")

    # EXACTLY ONE PIPELINE INSTANCE - NO RETRY FALLBACKS
    pipe = CogVideoXPipeline.from_pretrained(model_id, **kwargs)

    log_mem("After pipeline weights loaded")

    pipe = pipe.to(device)
    log_mem("After moving pipeline to CPU")

    if hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        log_mem("After enabling VAE slicing & tiling")

    print("[MEMORY] Pipeline loaded successfully")

if __name__ == "__main__":
    try:
        test_model_loading()
    except Exception as e:
        print(f"❌ [LOADING ERROR]: {e}")
        sys.exit(1)
