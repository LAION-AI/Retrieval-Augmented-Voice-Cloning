#!/usr/bin/env python3
"""
FastAPI server for Echo TTS generation.

Loads Echo TTS model on a specified GPU and serves generation requests.
Usage: python echo_tts_server.py --gpu 5 --port 9205
"""

import os
import sys
import argparse
import time
import tempfile

# Fix cuDNN library path issue
if "ml-general" in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = ""
# Disable torch dynamo to prevent tensordict import issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch.backends.cudnn.enabled = False

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

# Add Echo TTS source to path
sys.path.insert(0, os.environ.get("ECHO_TTS_SRC", os.path.expanduser("~/open-echo-tts/src")))

app = FastAPI(title="Echo TTS Server")

ECHO_TTS_SR = 44100


class State:
    pipeline = None
    device = None
    loading = False


state = State()


def save_wav(path, audio_tensor, sample_rate):
    """Save torch tensor as WAV using soundfile."""
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 1:
        pass
    elif audio_np.ndim == 2:
        audio_np = audio_np.T
    elif audio_np.ndim == 3:
        audio_np = audio_np.squeeze(0).T
    sf.write(path, audio_np, sample_rate, subtype="PCM_16")


def load_wav(path):
    """Load WAV to (tensor [1, samples], sr)."""
    audio_np, sr = sf.read(path, dtype="float32")
    if audio_np.ndim == 1:
        t = torch.from_numpy(audio_np).unsqueeze(0)
    else:
        t = torch.from_numpy(audio_np.T)
    return t, sr


def resample_tensor(audio_tensor, orig_sr, target_sr):
    """Resample using scipy."""
    if orig_sr == target_sr:
        return audio_tensor
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    audio_np = audio_tensor.cpu().numpy()
    resampled = resample_poly(audio_np, up, down, axis=-1).astype(np.float32)
    return torch.from_numpy(resampled)


def load_pipeline(device_str):
    """Load Echo TTS pipeline.

    We bypass open_echo_tts.__init__.py to avoid tensordict import issues
    by importing individual modules directly via importlib.
    """
    import importlib

    # Import submodules directly to avoid __init__.py chain that triggers tensordict
    loader_mod = importlib.import_module("open_echo_tts.pipeline.loader")
    tts_mod = importlib.import_module("open_echo_tts.pipeline.tts")

    load_model = loader_mod.load_model
    load_autoencoder = loader_mod.load_autoencoder
    load_pca_state = loader_mod.load_pca_state
    TTSPipeline = tts_mod.TTSPipeline

    print(f"Loading Echo TTS on {device_str}...", flush=True)
    model = load_model(device=device_str, delete_blockwise_modules=True)
    autoencoder = load_autoencoder(device=device_str)
    pca_state = load_pca_state(device=device_str)
    pipeline = TTSPipeline(model, autoencoder, pca_state)
    print("Echo TTS loaded!", flush=True)
    return pipeline


@app.on_event("startup")
async def startup():
    pass  # Lazy load on first request


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.pipeline is not None, "device": str(state.device)}


@app.post("/generate")
async def generate(
    text: str = Form(...),
    ref_audio_path: str = Form(...),
    seed: int = Form(0),
    num_steps: int = Form(40),
):
    """Generate speech from text with speaker reference audio.

    Args:
        text: Text to synthesize (will be prefixed with [S1] if not present)
        ref_audio_path: Path to reference speaker audio WAV (any sample rate)
        seed: Random seed
        num_steps: Number of diffusion steps

    Returns:
        JSON with output_path, duration, sample_rate, elapsed
    """
    try:
        # Lazy load
        if state.pipeline is None:
            if state.loading:
                for _ in range(240):
                    time.sleep(0.5)
                    if state.pipeline is not None:
                        break
                if state.pipeline is None:
                    return JSONResponse({"error": "Model still loading"}, status_code=503)
            else:
                state.loading = True
                try:
                    state.pipeline = load_pipeline(str(state.device))
                finally:
                    state.loading = False

        tts = state.pipeline

        # Configure steps
        from open_echo_tts.inference.sampler import SamplerConfig
        tts.config.sampler = SamplerConfig(num_steps=num_steps)

        # Load speaker reference audio and resample to 44.1kHz
        speaker_audio, speaker_sr = load_wav(ref_audio_path)
        if speaker_sr != ECHO_TTS_SR:
            speaker_audio = resample_tensor(speaker_audio, speaker_sr, ECHO_TTS_SR)

        # Normalize
        max_val = speaker_audio.abs().max().clamp(min=1.0)
        speaker_audio = speaker_audio / max_val

        # Prefix text
        gen_text = text if text.startswith("[S1]") else "[S1] " + text

        # Generate
        t0 = time.time()
        audio_out, norm_text = tts(
            gen_text,
            speaker_audio=speaker_audio.to(state.device),
            seed=seed,
        )
        elapsed = round(time.time() - t0, 3)

        # Save output
        output_dir = tempfile.mkdtemp(prefix="echo_tts_")
        output_path = os.path.join(output_dir, f"gen_s{seed}.wav")
        save_wav(output_path, audio_out[0].cpu(), ECHO_TTS_SR)

        duration = round(audio_out.shape[-1] / ECHO_TTS_SR, 3)

        return {
            "status": "ok",
            "output_path": output_path,
            "normalized_text": norm_text,
            "duration": duration,
            "sample_rate": ECHO_TTS_SR,
            "seed": seed,
            "elapsed": elapsed,
        }

    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    state.device = torch.device(f"cuda:{args.gpu}")
    print(f"Echo TTS server starting on GPU {args.gpu}, port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
