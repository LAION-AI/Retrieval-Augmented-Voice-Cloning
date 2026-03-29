"""
Dataset loader for emotion reference audio and LAION reference voices.

Handles:
- Downloading/caching emotion reference tars from HuggingFace
- Parsing samples (JSON metadata + NPY DACVAE latents) from tars
- Decoding DACVAE latents to WAV audio
- Downloading and extracting LAION clustered reference voices
"""

import io
import json
import os
import random
import tarfile
import tempfile

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.signal import resample_poly
from math import gcd

from config import (
    EMOTION_REF_DATASET, LAION_REF_VOICES, LAION_VOICES_DIR,
    DACVAE_WEIGHTS, DACVAE_SR, ECHO_TTS_SR, CHATTERBOX_SR,
    ALL_DIMENSIONS, TMP_DIR,
)


# ─── DACVAE Model (loaded once, on CPU) ─────────────────────────────────────

_dacvae_model = None

def get_dacvae():
    """Lazy-load DACVAE model on CPU."""
    global _dacvae_model
    if _dacvae_model is None:
        from dacvae import DACVAE
        _dacvae_model = DACVAE.load(DACVAE_WEIGHTS)
        _dacvae_model.eval()
        print(f"DACVAE loaded on CPU, sr={_dacvae_model.sample_rate}")
    return _dacvae_model


def decode_dacvae_npy(npy_data, return_tensor=False):
    """Decode DACVAE .npy latent to audio.

    Args:
        npy_data: numpy array of shape (frames, 128) fp16, or raw bytes
        return_tensor: if True return (tensor, sr), else return (np_array, sr)

    Returns:
        (audio, sample_rate) where audio is numpy array or tensor
    """
    dacvae = get_dacvae()

    if isinstance(npy_data, bytes):
        npy_data = np.load(io.BytesIO(npy_data))

    latent_t = torch.from_numpy(npy_data).float()
    # Shape: (frames, 128) -> (1, 128, frames)
    if latent_t.dim() == 2:
        latent_t = latent_t.T.unsqueeze(0)

    with torch.no_grad():
        audio = dacvae.decode(latent_t)

    sr = dacvae.sample_rate  # 48000
    if return_tensor:
        return audio.squeeze(0).cpu(), sr  # (1, samples), sr
    else:
        return audio.squeeze().cpu().numpy(), sr


def save_wav(path, audio, sr):
    """Save audio (numpy or tensor) as WAV."""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.ndim == 1:
        pass
    elif audio.ndim == 2:
        audio = audio.T  # (channels, samples) -> (samples, channels)
    elif audio.ndim == 3:
        audio = audio.squeeze(0).T
    sf.write(path, audio, sr, subtype="PCM_16")


def load_wav(path):
    """Load WAV file, return (tensor [1, samples], sr)."""
    audio_np, sr = sf.read(path, dtype="float32")
    if audio_np.ndim == 1:
        t = torch.from_numpy(audio_np).unsqueeze(0)
    else:
        t = torch.from_numpy(audio_np.T)
    return t, sr


def resample_audio(audio_tensor, orig_sr, target_sr):
    """Resample audio tensor using scipy."""
    if orig_sr == target_sr:
        return audio_tensor
    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    audio_np = audio_tensor.cpu().numpy()
    resampled = resample_poly(audio_np, up, down, axis=-1).astype(np.float32)
    return torch.from_numpy(resampled)


# ─── Emotion Reference Dataset ──────────────────────────────────────────────

def download_emotion_tar(dimension: str, bucket: tuple, cache_dir=None):
    """Download a specific emotion reference tar from HuggingFace.

    Returns local path to the tar file.
    """
    from config import bucket_to_tar_name
    tar_name = bucket_to_tar_name(dimension, bucket)
    tar_path = f"data/{tar_name}"

    try:
        local_path = hf_hub_download(
            repo_id=EMOTION_REF_DATASET,
            filename=tar_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return local_path
    except Exception as e:
        print(f"Warning: Could not download {tar_path}: {e}")
        return None


def parse_emotion_tar(tar_path):
    """Parse an emotion reference tar file.

    The tar contains per-sample:
      - {sample_id}.json  (metadata with all EI scores)
      - {sample_id}.target.npy  (DACVAE latent, shape (frames, 128) fp16)
      - {sample_id}.target.wav  (audio WAV file)

    Returns list of dicts, each containing:
        - 'json': parsed metadata dict (includes all EI scores)
        - 'npy': raw NPY bytes (DACVAE latent)
        - 'wav': raw WAV bytes (optional, if present)
        - 'sample_id': base name of the sample
    """
    samples = {}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()

            # Get filename (strip directory prefix)
            name = member.name
            if "/" in name:
                name = name.split("/", 1)[1]

            if name.endswith(".json"):
                base = name[:-5]  # strip .json
                samples.setdefault(base, {})["json"] = json.loads(data)
                samples[base]["sample_id"] = base
            elif name.endswith(".target.npy"):
                base = name[:-11]  # strip .target.npy
                samples.setdefault(base, {})["npy"] = data
                samples[base]["sample_id"] = base
            elif name.endswith(".npy"):
                base = name[:-4]  # strip .npy
                samples.setdefault(base, {})["npy"] = data
                samples[base]["sample_id"] = base
            elif name.endswith(".target.wav"):
                base = name[:-11]  # strip .target.wav
                samples.setdefault(base, {})["wav"] = data
                samples[base]["sample_id"] = base
            elif name.endswith(".wav"):
                base = name[:-4]
                samples.setdefault(base, {})["wav"] = data
                samples[base]["sample_id"] = base

    # Return samples that have at least json + (npy or wav)
    result = [s for s in samples.values()
              if "json" in s and ("npy" in s or "wav" in s)]
    return result


def get_emotion_samples(dimension, bucket, cache_dir=None):
    """Download and parse emotion reference samples for a dimension+bucket.

    Returns list of sample dicts.
    """
    tar_path = download_emotion_tar(dimension, bucket, cache_dir=cache_dir)
    if tar_path is None:
        return []
    return parse_emotion_tar(tar_path)


def decode_sample_to_wav(sample, output_dir, target_sr=None):
    """Decode a sample to WAV file.

    Supports both:
    - Direct WAV data (from 'wav' key)
    - DACVAE latent decode (from 'npy' key)

    Args:
        sample: dict with 'wav' and/or 'npy' key
        output_dir: directory to save WAV
        target_sr: if set, resample to this rate

    Returns:
        path to saved WAV file, and sample rate
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_id = sample.get("sample_id", "unknown")
    wav_path = os.path.join(output_dir, f"{sample_id}.wav")

    if "wav" in sample:
        # Direct WAV - write it out, then optionally resample
        with open(wav_path, "wb") as f:
            f.write(sample["wav"])
        audio_np, sr = sf.read(wav_path, dtype="float32")

        if target_sr and target_sr != sr:
            audio_t = torch.from_numpy(audio_np).unsqueeze(0) if audio_np.ndim == 1 \
                else torch.from_numpy(audio_np.T)
            audio_t = resample_audio(audio_t, sr, target_sr)
            audio_np = audio_t.squeeze(0).numpy() if audio_t.shape[0] == 1 \
                else audio_t.numpy().T
            sr = target_sr
            sf.write(wav_path, audio_np, sr, subtype="PCM_16")

        return wav_path, sr

    elif "npy" in sample:
        # DACVAE decode
        audio_np, sr = decode_dacvae_npy(sample["npy"])

        if target_sr and target_sr != sr:
            audio_t = torch.from_numpy(audio_np).unsqueeze(0)
            audio_t = resample_audio(audio_t, sr, target_sr)
            audio_np = audio_t.squeeze(0).numpy()
            sr = target_sr

        sf.write(wav_path, audio_np, sr, subtype="PCM_16")
        return wav_path, sr
    else:
        raise ValueError(f"Sample {sample_id} has neither 'wav' nor 'npy' data")


# ─── LAION Reference Voices ─────────────────────────────────────────────────

_laion_voices_cache = None

def download_laion_voices():
    """Download and extract LAION clustered reference voices."""
    if os.path.exists(LAION_VOICES_DIR) and len(os.listdir(LAION_VOICES_DIR)) > 10:
        print(f"LAION voices already downloaded: {LAION_VOICES_DIR}")
        return

    os.makedirs(LAION_VOICES_DIR, exist_ok=True)
    print("Downloading LAION clustered reference voices...")

    # Download audio.tar.gz
    tar_path = hf_hub_download(
        repo_id=LAION_REF_VOICES,
        filename="audio.tar.gz",
        repo_type="dataset",
    )

    # Extract
    print(f"Extracting to {LAION_VOICES_DIR}...")
    import subprocess
    subprocess.run(
        ["tar", "xzf", tar_path, "-C", LAION_VOICES_DIR],
        check=True,
    )
    print(f"Extracted LAION reference voices")


def get_laion_voice_paths():
    """Get list of all LAION reference voice file paths."""
    global _laion_voices_cache
    if _laion_voices_cache is not None:
        return _laion_voices_cache

    paths = []
    for root, dirs, files in os.walk(LAION_VOICES_DIR):
        for f in files:
            if f.endswith((".mp3", ".wav", ".flac")):
                paths.append(os.path.join(root, f))

    _laion_voices_cache = sorted(paths)
    return _laion_voices_cache


def get_random_laion_voice():
    """Get a random LAION reference voice path."""
    paths = get_laion_voice_paths()
    if not paths:
        raise RuntimeError(f"No LAION reference voices found in {LAION_VOICES_DIR}")
    return random.choice(paths)


# ─── Utilities ───────────────────────────────────────────────────────────────

def get_audio_duration(wav_path):
    """Get duration in seconds of a WAV file."""
    info = sf.info(wav_path)
    return info.duration


def get_all_dimension_buckets():
    """Yield (dimension, bucket) for all dimensions and their buckets (from config)."""
    for dim_name, dim_info in ALL_DIMENSIONS.items():
        for bucket in dim_info["buckets"]:
            yield dim_name, bucket


def get_all_available_dimension_buckets():
    """Yield (dimension, bucket) for all actually available tars in HF dataset."""
    from config import discover_available_tars
    tars = discover_available_tars()
    for dim, bucket_strs in sorted(tars.items()):
        for bs in sorted(bucket_strs):
            parts = bs.split('to')
            try:
                bmin = int(parts[0])
                bmax = int(parts[1])
                yield dim, (bmin, bmax)
            except ValueError:
                continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-refs", action="store_true",
                        help="Download LAION reference voices")
    parser.add_argument("--test-bucket", type=str, default=None,
                        help="Test loading a bucket, e.g. 'Anger_2to3'")
    args = parser.parse_args()

    if args.download_refs:
        download_laion_voices()
        paths = get_laion_voice_paths()
        print(f"Found {len(paths)} reference voice files")

    if args.test_bucket:
        # Parse dimension_minTomax
        parts = args.test_bucket.rsplit("_", 1)
        dim = parts[0]
        bmin, bmax = parts[1].split("to")
        bucket = (int(bmin), int(bmax))
        print(f"Testing {dim} bucket {bucket}...")
        samples = get_emotion_samples(dim, bucket)
        print(f"Got {len(samples)} samples")
        if samples:
            print(f"First sample keys: {list(samples[0]['json'].keys())[:10]}...")
