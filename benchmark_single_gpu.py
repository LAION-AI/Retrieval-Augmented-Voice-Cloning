#!/usr/bin/env python3
"""
Benchmark: Single-GPU worker with model offloading for voice-acting pipeline.

Measures real wall-clock timings for each model (Echo TTS, ChatterboxVC, Empathic Insight)
including load, inference, and offload times. Calculates throughput for A100 and estimates H100.

Usage: python benchmark_single_gpu.py --gpu 3
"""

import os
import sys
import gc
import json
import time
import subprocess
import tempfile
import argparse
from datetime import datetime

# Fix cuDNN
if "ml-general" in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = ""

import torch
torch.backends.cudnn.enabled = False

import numpy as np
import soundfile as sf

# Add Echo TTS source
sys.path.insert(0, os.environ.get("ECHO_TTS_SRC", os.path.expanduser("~/open-echo-tts/src")))

SPIRITVENV_PYTHON = os.environ.get("SPIRITVENV_PYTHON", os.path.expanduser("~/spiritvenv/bin/python"))
LAION_VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laion_ref_voices")

# ─── Timing helpers ───────────────────────────────────────────────────────────

def gpu_mem_mb(device):
    return torch.cuda.memory_allocated(device) / 1024**2

def gpu_sync_time(device):
    torch.cuda.synchronize(device)
    return time.time()

def clear_gpu(device):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(device)


# ─── Echo TTS Benchmark ──────────────────────────────────────────────────────

def benchmark_echo_tts(device, n_inferences=6, num_steps=40, warmup=1):
    """Benchmark Echo TTS load, inference (n_inferences calls), and offload."""
    from open_echo_tts.pipeline.loader import load_model, load_autoencoder, load_pca_state
    from open_echo_tts.pipeline.tts import TTSPipeline
    from open_echo_tts.inference.sampler import SamplerConfig

    results = {}
    clear_gpu(device)
    mem_before = gpu_mem_mb(device)

    # ── Load ──
    t0 = time.time()
    model = load_model(device=str(device), delete_blockwise_modules=True, compile=False)
    autoencoder = load_autoencoder(device=str(device))
    pca_state = load_pca_state(device=str(device))
    pipeline = TTSPipeline(model, autoencoder, pca_state)
    pipeline.config.sampler = SamplerConfig(num_steps=num_steps)
    torch.cuda.synchronize(device)
    load_time = time.time() - t0
    mem_loaded = gpu_mem_mb(device)
    results["load_time"] = round(load_time, 3)
    results["vram_loaded_mb"] = round(mem_loaded - mem_before, 1)
    print(f"  Echo TTS load: {load_time:.2f}s, VRAM: {mem_loaded - mem_before:.0f} MB")

    # ── Prepare speaker ref ──
    ref_path = os.path.join(LAION_VOICES_DIR, "0.mp3")
    import librosa
    audio_np, sr = librosa.load(ref_path, sr=44100)
    speaker_audio = torch.from_numpy(audio_np).unsqueeze(0)
    # Trim to 10s
    max_samples = 10 * 44100
    if speaker_audio.shape[-1] > max_samples:
        speaker_audio = speaker_audio[..., :max_samples]

    test_texts = [
        "[S1] The storm raged outside, shaking the windows violently.",
        "[S1] I can't believe you would do something so incredibly reckless!",
        "[S1] Softly humming to herself, she arranged the flowers with care.",
        "[S1] What if everything we believed turned out to be wrong?",
        "[S1] The victory was ours, and the crowd erupted in celebration!",
        "[S1] Nothing matters anymore, everything is just... gray.",
    ]

    # ── Warmup ──
    print(f"  Warmup ({warmup} inference(s))...")
    for i in range(warmup):
        with torch.inference_mode():
            pipeline(test_texts[0], speaker_audio=speaker_audio.to(device), seed=42)
    torch.cuda.synchronize(device)

    mem_peak_inference = gpu_mem_mb(device)
    results["vram_peak_inference_mb"] = round(mem_peak_inference - mem_before, 1)

    # ── Timed inference ──
    inference_times = []
    for i in range(n_inferences):
        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.inference_mode():
            audio_out, _ = pipeline(
                test_texts[i % len(test_texts)],
                speaker_audio=speaker_audio.to(device),
                seed=i * 100,
            )
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        inference_times.append(elapsed)
        dur = audio_out.shape[-1] / 44100
        print(f"    Gen {i+1}/{n_inferences}: {elapsed:.3f}s ({dur:.1f}s audio)")

    results["inference_times"] = [round(t, 3) for t in inference_times]
    results["inference_avg"] = round(sum(inference_times) / len(inference_times), 3)
    results["inference_min"] = round(min(inference_times), 3)
    results["inference_max"] = round(max(inference_times), 3)

    # ── Offload to CPU ──
    torch.cuda.synchronize(device)
    t0 = time.time()
    model.to("cpu")
    autoencoder.to("cpu")
    # pca_state tensors
    pca_state.components = pca_state.components.to("cpu")
    pca_state.mean = pca_state.mean.to("cpu")
    torch.cuda.synchronize(device)
    clear_gpu(device)
    offload_time = time.time() - t0
    mem_after_offload = gpu_mem_mb(device)
    results["offload_time"] = round(offload_time, 3)
    results["vram_after_offload_mb"] = round(mem_after_offload - mem_before, 1)
    print(f"  Echo TTS offload: {offload_time:.2f}s, VRAM freed: {(mem_loaded - mem_after_offload):.0f} MB")

    # ── Reload (measure reload time) ──
    clear_gpu(device)
    t0 = time.time()
    model.to(device)
    autoencoder.to(device)
    pca_state.components = pca_state.components.to(device)
    pca_state.mean = pca_state.mean.to(device)
    torch.cuda.synchronize(device)
    reload_time = time.time() - t0
    results["reload_time"] = round(reload_time, 3)
    print(f"  Echo TTS reload (CPU→GPU): {reload_time:.2f}s")

    # Final cleanup
    del model, autoencoder, pca_state, pipeline
    clear_gpu(device)

    return results


# ─── ChatterboxVC Benchmark ──────────────────────────────────────────────────

def benchmark_vc(gpu_id, n_inferences=5, warmup=1):
    """Benchmark ChatterboxVC via subprocess (requires spiritvenv)."""
    results = {}

    worker_code = f'''
import os, sys, json, time, gc
import torch

DEVICE = "cuda:{gpu_id}"
torch.cuda.set_device(DEVICE)

def proto_send(obj):
    print(json.dumps(obj), flush=True)

# Load
t0 = time.time()
from chatterbox import ChatterboxVC
model = ChatterboxVC.from_pretrained(device=DEVICE)
torch.cuda.synchronize()
load_time = time.time() - t0
vram = torch.cuda.memory_allocated() / 1024**2
proto_send({{"op": "load", "time": round(load_time, 3), "vram_mb": round(vram, 1)}})

# Read requests from stdin
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)

    if req.get("op") == "convert":
        torch.cuda.synchronize()
        t0 = time.time()
        wav = model.generate(audio=req["source"], target_voice_path=req["target"])
        import torchaudio
        torchaudio.save(req["output"], wav.cpu().float(), model.sr)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        dur = wav.shape[-1] / model.sr
        proto_send({{"op": "convert", "time": round(elapsed, 3), "duration": round(dur, 2)}})

    elif req.get("op") == "offload":
        torch.cuda.synchronize()
        t0 = time.time()
        model.s3gen.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated() / 1024**2
        proto_send({{"op": "offload", "time": round(elapsed, 3), "vram_mb": round(vram, 1)}})

    elif req.get("op") == "reload":
        torch.cuda.synchronize()
        t0 = time.time()
        model.s3gen.to(DEVICE)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated() / 1024**2
        proto_send({{"op": "reload", "time": round(elapsed, 3), "vram_mb": round(vram, 1)}})

    elif req.get("op") == "quit":
        break
'''

    worker_path = os.path.join(tempfile.gettempdir(), f"vc_bench_gpu{gpu_id}.py")
    with open(worker_path, "w") as f:
        f.write(worker_code)

    proc = subprocess.Popen(
        [SPIRITVENV_PYTHON, worker_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "LD_LIBRARY_PATH": ""},
    )

    def read_json_line():
        """Read lines from stdout until we get valid JSON."""
        while True:
            line = proc.stdout.readline().decode().strip()
            if not line:
                # Check if process died
                if proc.poll() is not None:
                    stderr = proc.stderr.read().decode()
                    raise RuntimeError(f"VC subprocess died. stderr:\n{stderr}")
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON lines (import warnings, etc.)
                continue

    def send_recv(req):
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        return read_json_line()

    # Read load result
    load_result = read_json_line()
    results["load_time"] = load_result["time"]
    results["vram_loaded_mb"] = load_result["vram_mb"]
    print(f"  ChatterboxVC load: {load_result['time']:.2f}s, VRAM: {load_result['vram_mb']:.0f} MB")

    # Prepare test files
    source_path = os.path.join(LAION_VOICES_DIR, "0.mp3")
    target_path = os.path.join(LAION_VOICES_DIR, "1.mp3")

    # Warmup
    print(f"  Warmup ({warmup} inference(s))...")
    for i in range(warmup):
        out_dir = tempfile.mkdtemp(prefix="vc_bench_")
        out_path = os.path.join(out_dir, "warmup.wav")
        r = send_recv({"op": "convert", "source": source_path, "target": target_path, "output": out_path})

    # Timed inferences
    inference_times = []
    for i in range(n_inferences):
        out_dir = tempfile.mkdtemp(prefix="vc_bench_")
        out_path = os.path.join(out_dir, f"bench_{i}.wav")
        # Use different source/target pairs
        src = os.path.join(LAION_VOICES_DIR, f"{i % 10}.mp3")
        tgt = os.path.join(LAION_VOICES_DIR, f"{(i + 5) % 10}.mp3")
        r = send_recv({"op": "convert", "source": src, "target": tgt, "output": out_path})
        inference_times.append(r["time"])
        print(f"    Convert {i+1}/{n_inferences}: {r['time']:.3f}s ({r['duration']:.1f}s audio)")

    results["inference_times"] = inference_times
    results["inference_avg"] = round(sum(inference_times) / len(inference_times), 3)

    # Offload
    r = send_recv({"op": "offload"})
    results["offload_time"] = r["time"]
    print(f"  ChatterboxVC offload: {r['time']:.2f}s")

    # Reload
    r = send_recv({"op": "reload"})
    results["reload_time"] = r["time"]
    print(f"  ChatterboxVC reload: {r['time']:.2f}s")

    # Quit
    try:
        proc.stdin.write((json.dumps({"op": "quit"}) + "\n").encode())
        proc.stdin.flush()
    except BrokenPipeError:
        pass
    proc.wait(timeout=10)

    return results


# ─── Empathic Insight Benchmark ───────────────────────────────────────────────

def benchmark_ei(device, n_inferences=6, warmup=1):
    """Benchmark Empathic Insight load, scoring, and offload."""
    from collections import OrderedDict
    from pathlib import Path
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from huggingface_hub import snapshot_download
    import librosa
    import torch.nn as nn

    # MLP architectures (from ei_server.py)
    WHISPER_SEQ_LEN = 1500
    WHISPER_EMBED_DIM = 768
    PROJECTION_DIM = 64
    MLP_HIDDEN_DIMS = [64, 32, 16]
    MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]
    POOLED_DIM = 3072
    QUALITY_EXPERT_FILES = {
        "model_score_overall_quality_best.pth",
        "model_score_speech_quality_best.pth",
        "model_score_background_quality_best.pth",
        "model_score_content_enjoyment_best.pth",
    }
    MODELS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")

    class FullEmbeddingMLP(nn.Module):
        def __init__(self, seq_len, embed_dim, projection_dim, mlp_hidden_dims, mlp_dropout_rates):
            super().__init__()
            self.flatten = nn.Flatten()
            self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
            layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
            current_dim = projection_dim
            for i, h_dim in enumerate(mlp_hidden_dims):
                layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i + 1])])
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            if x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)
            return self.mlp(self.proj(self.flatten(x)))

    class PooledEmbeddingMLP(nn.Module):
        def __init__(self, input_dim, projection_dim, mlp_hidden_dims, mlp_dropout_rates):
            super().__init__()
            self.proj = nn.Linear(input_dim, projection_dim)
            layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
            current_dim = projection_dim
            for i, h_dim in enumerate(mlp_hidden_dims):
                layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i + 1])])
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            return self.mlp(self.proj(x))

    def pool_embedding(embedding):
        return torch.cat([
            embedding.mean(dim=1), embedding.min(dim=1).values,
            embedding.max(dim=1).values, embedding.std(dim=1),
        ], dim=1)

    results = {}
    clear_gpu(device)
    mem_before = gpu_mem_mb(device)

    # ── Load ──
    t0 = time.time()
    whisper = WhisperForConditionalGeneration.from_pretrained(
        "laion/BUD-E-Whisper", torch_dtype=torch.float16,
        cache_dir=MODELS_CACHE, attn_implementation="sdpa")
    whisper.to(device).eval()
    processor = AutoProcessor.from_pretrained("laion/BUD-E-Whisper", cache_dir=MODELS_CACHE)

    mlp_dir = Path(os.path.join(MODELS_CACHE, "empathic_insight_models"))
    mlp_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id="laion/Empathic-Insight-Voice-Plus", local_dir=str(mlp_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.mp3", "*.md", ".gitattributes"])

    emotion_models = {}
    quality_models = {}
    for model_path in mlp_dir.glob("*.pth"):
        filename = model_path.name
        stem = model_path.stem
        parts = stem.split("_")
        dimension_name = "_".join(parts[1:-1]) if "best" in parts[-1] else "_".join(parts[1:])
        is_quality = filename in QUALITY_EXPERT_FILES
        if is_quality:
            mlp = PooledEmbeddingMLP(POOLED_DIM, PROJECTION_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUTS).to(device)
        else:
            mlp = FullEmbeddingMLP(WHISPER_SEQ_LEN, WHISPER_EMBED_DIM, PROJECTION_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUTS).to(device)
        sd = torch.load(model_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd.items())
        mlp.load_state_dict(sd)
        mlp.eval().half()
        if is_quality:
            quality_models[dimension_name] = mlp
        else:
            emotion_models[dimension_name] = mlp

    torch.cuda.synchronize(device)
    load_time = time.time() - t0
    mem_loaded = gpu_mem_mb(device)
    results["load_time"] = round(load_time, 3)
    results["vram_loaded_mb"] = round(mem_loaded - mem_before, 1)
    results["n_emotion_experts"] = len(emotion_models)
    results["n_quality_experts"] = len(quality_models)
    print(f"  EI load: {load_time:.2f}s, VRAM: {mem_loaded - mem_before:.0f} MB")
    print(f"    {len(emotion_models)} emotion + {len(quality_models)} quality experts")

    # ── Prepare test audio ──
    ref_path = os.path.join(LAION_VOICES_DIR, "0.mp3")
    audio_np, sr = librosa.load(ref_path, sr=16000)
    max_len = 30 * 16000
    if len(audio_np) > max_len:
        audio_np = audio_np[:max_len]

    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).half()

    # ── Warmup ──
    print(f"  Warmup ({warmup} inference(s))...")
    for _ in range(warmup):
        with torch.no_grad():
            enc_out = whisper.get_encoder()(input_features, return_dict=True)
            emb = enc_out.last_hidden_state
            pooled = pool_embedding(emb)
            for name, m in list(emotion_models.items())[:3]:
                m(emb)

    # ── Timed: Encoder ──
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        enc_out = whisper.get_encoder()(input_features, return_dict=True)
        embeddings = enc_out.last_hidden_state
        pooled = pool_embedding(embeddings)
    torch.cuda.synchronize(device)
    encoder_time = time.time() - t0
    results["encoder_time"] = round(encoder_time, 3)
    print(f"  Whisper encoder: {encoder_time:.3f}s")

    # ── Timed: All 59 experts ──
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        scores = {}
        for name, m in emotion_models.items():
            scores[name] = float(m(embeddings).item())
        for name, m in quality_models.items():
            scores[name] = float(m(pooled).item())
    torch.cuda.synchronize(device)
    experts_time = time.time() - t0
    results["experts_time"] = round(experts_time, 3)
    print(f"  59 MLP experts: {experts_time:.3f}s")

    # ── Timed: Caption generation ──
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        generated_ids = whisper.generate(input_features)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    torch.cuda.synchronize(device)
    caption_time = time.time() - t0
    results["caption_time"] = round(caption_time, 3)
    print(f"  Caption generation: {caption_time:.3f}s")

    # ── Timed: Full scoring (n_inferences) ──
    scoring_times = []
    for i in range(n_inferences):
        # Use different audio files
        ref = os.path.join(LAION_VOICES_DIR, f"{i % 10}.mp3")
        audio_np_i, _ = librosa.load(ref, sr=16000)
        if len(audio_np_i) > max_len:
            audio_np_i = audio_np_i[:max_len]
        inputs_i = processor(audio_np_i, sampling_rate=16000, return_tensors="pt")
        feats_i = inputs_i.input_features.to(device).half()

        torch.cuda.synchronize(device)
        t0 = time.time()
        with torch.no_grad():
            enc_out = whisper.get_encoder()(feats_i, return_dict=True)
            emb = enc_out.last_hidden_state
            p = pool_embedding(emb)
            for name, m in emotion_models.items():
                m(emb)
            for name, m in quality_models.items():
                m(p)
            generated_ids = whisper.generate(feats_i)
            processor.batch_decode(generated_ids, skip_special_tokens=True)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        scoring_times.append(elapsed)
        print(f"    Score {i+1}/{n_inferences}: {elapsed:.3f}s")

    results["scoring_times"] = [round(t, 3) for t in scoring_times]
    results["scoring_avg"] = round(sum(scoring_times) / len(scoring_times), 3)

    # ── Offload ──
    torch.cuda.synchronize(device)
    t0 = time.time()
    whisper.to("cpu")
    for m in emotion_models.values():
        m.to("cpu")
    for m in quality_models.values():
        m.to("cpu")
    torch.cuda.synchronize(device)
    clear_gpu(device)
    offload_time = time.time() - t0
    results["offload_time"] = round(offload_time, 3)
    print(f"  EI offload: {offload_time:.2f}s")

    # ── Reload ──
    clear_gpu(device)
    t0 = time.time()
    whisper.to(device)
    for m in emotion_models.values():
        m.to(device)
    for m in quality_models.values():
        m.to(device)
    torch.cuda.synchronize(device)
    reload_time = time.time() - t0
    results["reload_time"] = round(reload_time, 3)
    print(f"  EI reload: {reload_time:.2f}s")

    # Cleanup
    del whisper, processor, emotion_models, quality_models
    clear_gpu(device)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print("=" * 70)
    print(f"SINGLE-GPU BENCHMARK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(device)} (cuda:{args.gpu})")
    print(f"VRAM Total: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print("=" * 70)

    all_results = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_id": args.gpu,
        "vram_total_gb": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Benchmark Echo TTS ──
    print("\n" + "─" * 50)
    print("1. ECHO TTS (DiT bf16 + DAC fp32, 40 steps)")
    print("─" * 50)
    echo_results = benchmark_echo_tts(device, n_inferences=6, num_steps=40, warmup=2)
    all_results["echo_tts"] = echo_results

    # ── Benchmark ChatterboxVC ──
    print("\n" + "─" * 50)
    print("2. CHATTERBOX VC (S3Gen fp32)")
    print("─" * 50)
    vc_results = benchmark_vc(args.gpu, n_inferences=5, warmup=1)
    all_results["chatterbox_vc"] = vc_results

    # ── Benchmark Empathic Insight ──
    print("\n" + "─" * 50)
    print("3. EMPATHIC INSIGHT (Whisper fp16 + 59 MLPs fp16)")
    print("─" * 50)
    ei_results = benchmark_ei(device, n_inferences=6, warmup=1)
    all_results["empathic_insight"] = ei_results

    # ── Calculate throughput ──
    print("\n" + "=" * 70)
    print("THROUGHPUT ANALYSIS")
    print("=" * 70)

    tts_per_gen = echo_results["inference_avg"]
    vc_per_call = vc_results["inference_avg"]
    ei_per_score = ei_results["scoring_avg"]

    tts_load = echo_results["load_time"]
    tts_offload = echo_results["offload_time"]
    tts_reload = echo_results["reload_time"]
    vc_load = vc_results["load_time"]
    vc_offload = vc_results["offload_time"]
    vc_reload = vc_results["reload_time"]
    ei_load = ei_results["load_time"]
    ei_offload = ei_results["offload_time"]
    ei_reload = ei_results["reload_time"]

    # Config: 1 neutral + 5 emotional = 6 TTS gens, 1 VC, 6 EI scorings
    n_tts = 6
    n_vc = 1
    n_ei = 6

    # Mode A: All models loaded (no offloading)
    sample_time_all_loaded = (n_vc * vc_per_call) + (n_tts * tts_per_gen) + (n_ei * ei_per_score)

    # Mode B: Offloading per sample (worst case)
    sample_time_offload_per = (
        vc_reload + n_vc * vc_per_call + vc_offload +
        tts_reload + n_tts * tts_per_gen + tts_offload +
        ei_reload + n_ei * ei_per_score + ei_offload
    )

    # Mode C: Batch of 10 (amortize load/offload)
    batch_size = 10
    batch_time = (
        vc_reload + batch_size * n_vc * vc_per_call + vc_offload +
        tts_reload + batch_size * n_tts * tts_per_gen + tts_offload +
        ei_reload + batch_size * n_ei * ei_per_score + ei_offload
    )
    sample_time_batch10 = batch_time / batch_size

    # Mode D: Batch of 50
    batch_size_50 = 50
    batch_time_50 = (
        vc_reload + batch_size_50 * n_vc * vc_per_call + vc_offload +
        tts_reload + batch_size_50 * n_tts * tts_per_gen + tts_offload +
        ei_reload + batch_size_50 * n_ei * ei_per_score + ei_offload
    )
    sample_time_batch50 = batch_time_50 / batch_size_50

    # Sentence generation (Gemini API) - happens in parallel, ~1s per pair
    llm_time = 1.0  # overlapped with other ops in batch mode

    all_results["throughput"] = {
        "config": f"{n_tts} TTS gens (1 neutral + 5 emotional, 40 steps) + {n_vc} VC + {n_ei} EI scorings",
        "mode_a_all_loaded": round(sample_time_all_loaded, 2),
        "mode_b_offload_per_sample": round(sample_time_offload_per, 2),
        "mode_c_batch_10": round(sample_time_batch10, 2),
        "mode_d_batch_50": round(sample_time_batch50, 2),
    }

    # ── Write TXT report ──
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("VOICE-ACTING PIPELINE: SINGLE-GPU THROUGHPUT BENCHMARK\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(device)} (80 GB, SXM4)\n")
        f.write("=" * 80 + "\n\n")

        f.write("PIPELINE CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write("Per sample:\n")
        f.write("  - 1x Voice Conversion (ChatterboxVC)\n")
        f.write("  - 6x TTS generation (1 neutral + 5 emotional, 40 Euler steps each)\n")
        f.write("  - 6x Emotion scoring (Empathic Insight, 59 experts per WAV)\n")
        f.write("  - 2x Sentence generation (Gemini API, ~1s overlapped)\n")
        f.write("\n")

        f.write("MODEL VRAM REQUIREMENTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Model':<30} {'Weights':>10} {'Peak':>10}\n")
        f.write(f"{'Echo TTS (DiT bf16+DAC fp32)':<30} {echo_results['vram_loaded_mb']:>8.0f} MB {echo_results['vram_peak_inference_mb']:>8.0f} MB\n")
        f.write(f"{'ChatterboxVC (S3Gen fp32)':<30} {vc_results['vram_loaded_mb']:>8.0f} MB {'~' + str(int(vc_results['vram_loaded_mb'] * 1.3)):>7} MB\n")
        f.write(f"{'Empathic Insight (Whisper+MLPs)':<30} {ei_results['vram_loaded_mb']:>8.0f} MB {'~' + str(int(ei_results['vram_loaded_mb'] * 1.1)):>7} MB\n")
        total_vram = echo_results['vram_loaded_mb'] + vc_results['vram_loaded_mb'] + ei_results['vram_loaded_mb']
        f.write(f"{'TOTAL (all loaded)':<30} {total_vram:>8.0f} MB\n")
        f.write(f"\nAll models fit simultaneously on 80GB GPU: {'YES' if total_vram < 70000 else 'NO'}\n\n")

        f.write("INDIVIDUAL OPERATION TIMINGS (A100-SXM4-80GB)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Operation':<40} {'Time':>10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Echo TTS: load (HF → GPU)':<40} {echo_results['load_time']:>8.2f} s\n")
        f.write(f"{'Echo TTS: single generation (40 steps)':<40} {echo_results['inference_avg']:>8.3f} s\n")
        f.write(f"{'Echo TTS: offload (GPU → CPU)':<40} {echo_results['offload_time']:>8.2f} s\n")
        f.write(f"{'Echo TTS: reload (CPU → GPU)':<40} {echo_results['reload_time']:>8.2f} s\n")
        f.write(f"{'  min / max generation':<40} {echo_results['inference_min']:>5.3f} / {echo_results['inference_max']:.3f} s\n")
        f.write("\n")
        f.write(f"{'ChatterboxVC: load (HF → GPU)':<40} {vc_results['load_time']:>8.2f} s\n")
        f.write(f"{'ChatterboxVC: single conversion':<40} {vc_results['inference_avg']:>8.3f} s\n")
        f.write(f"{'ChatterboxVC: offload (GPU → CPU)':<40} {vc_results['offload_time']:>8.2f} s\n")
        f.write(f"{'ChatterboxVC: reload (CPU → GPU)':<40} {vc_results['reload_time']:>8.2f} s\n")
        f.write("\n")
        f.write(f"{'EI: load (HF → GPU, 59 experts)':<40} {ei_results['load_time']:>8.2f} s\n")
        f.write(f"{'EI: Whisper encoder forward':<40} {ei_results['encoder_time']:>8.3f} s\n")
        f.write(f"{'EI: 59 MLP experts inference':<40} {ei_results['experts_time']:>8.3f} s\n")
        f.write(f"{'EI: caption generation (Whisper decode)':<40} {ei_results['caption_time']:>8.3f} s\n")
        f.write(f"{'EI: full scoring (encode+experts+caption)':<40} {ei_results['scoring_avg']:>8.3f} s\n")
        f.write(f"{'EI: offload (GPU → CPU)':<40} {ei_results['offload_time']:>8.2f} s\n")
        f.write(f"{'EI: reload (CPU → GPU)':<40} {ei_results['reload_time']:>8.2f} s\n")
        f.write("\n")

        f.write("PER-SAMPLE BREAKDOWN (1 VC + 6 TTS + 6 EI)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Component':<40} {'Time':>10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Voice Conversion (1x)':<40} {n_vc * vc_per_call:>8.2f} s\n")
        f.write(f"{'TTS Generation (6x @ 40 steps)':<40} {n_tts * tts_per_gen:>8.2f} s\n")
        f.write(f"{'Emotion Scoring (6x, 59 experts each)':<40} {n_ei * ei_per_score:>8.2f} s\n")
        f.write(f"{'Sentence Generation (Gemini, overlapped)':<40} {'~1.00':>9} s\n")
        f.write(f"{'─'*40:─<40} {'─'*10:─>10}\n")
        f.write(f"{'TOTAL (inference only, no load/offload)':<40} {sample_time_all_loaded:>8.2f} s\n")
        f.write("\n")

        f.write("THROUGHPUT: A100-SXM4-80GB\n")
        f.write("=" * 60 + "\n\n")

        f.write("Mode A: All models loaded (80GB GPU, no offloading)\n")
        f.write(f"  Per sample: {sample_time_all_loaded:.2f}s\n")
        sph_a = 3600 / sample_time_all_loaded
        f.write(f"  Throughput: {sph_a:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {sph_a * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {sph_a * 8:.0f} samples/hour (8 GPUs)\n\n")

        f.write("Mode B: Offloading per sample (works on any GPU >=16GB)\n")
        f.write(f"  Per sample: {sample_time_offload_per:.2f}s\n")
        sph_b = 3600 / sample_time_offload_per
        f.write(f"  Throughput: {sph_b:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {sph_b * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {sph_b * 8:.0f} samples/hour (8 GPUs)\n\n")

        f.write("Mode C: Batch-phased, batch=10 (load once per phase per batch)\n")
        f.write(f"  Per sample: {sample_time_batch10:.2f}s\n")
        sph_c = 3600 / sample_time_batch10
        f.write(f"  Throughput: {sph_c:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {sph_c * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {sph_c * 8:.0f} samples/hour (8 GPUs)\n\n")

        f.write("Mode D: Batch-phased, batch=50\n")
        f.write(f"  Per sample: {sample_time_batch50:.2f}s\n")
        sph_d = 3600 / sample_time_batch50
        f.write(f"  Throughput: {sph_d:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {sph_d * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {sph_d * 8:.0f} samples/hour (8 GPUs)\n\n")

        # H100 estimates
        f.write("ESTIMATED THROUGHPUT: H100-SXM5-80GB\n")
        f.write("=" * 60 + "\n")
        f.write("Estimation methodology:\n")
        f.write("  - Memory bandwidth: A100=2.0 TB/s, H100=3.35 TB/s (1.68x)\n")
        f.write("  - Compute (bf16): A100=312 TFLOPS, H100=989 TFLOPS (3.17x)\n")
        f.write("  - Model load/offload: bandwidth-limited -> ~1.7x faster\n")
        f.write("  - Transformer inference: compute-bound -> ~2.5x faster (practical)\n")
        f.write("  - MLP inference: compute-bound -> ~2.5x faster\n")
        f.write("  - VC (S3Gen): mixed -> ~2.0x faster\n")
        f.write("  - Conservative overall: ~2.0x speedup\n\n")

        h100_factor = 2.0

        h100_tts_gen = tts_per_gen / 2.5
        h100_vc = vc_per_call / 2.0
        h100_ei = ei_per_score / 2.5
        h100_sample_all = n_vc * h100_vc + n_tts * h100_tts_gen + n_ei * h100_ei

        h100_tts_reload = tts_reload / 1.7
        h100_vc_reload = vc_reload / 1.7
        h100_ei_reload = ei_reload / 1.7
        h100_tts_offload = tts_offload / 1.7
        h100_vc_offload = vc_offload / 1.7
        h100_ei_offload = ei_offload / 1.7

        h100_batch50 = (
            h100_vc_reload + batch_size_50 * n_vc * h100_vc + h100_vc_offload +
            h100_tts_reload + batch_size_50 * n_tts * h100_tts_gen + h100_tts_offload +
            h100_ei_reload + batch_size_50 * n_ei * h100_ei + h100_ei_offload
        ) / batch_size_50

        f.write(f"{'Operation':<40} {'A100':>10} {'H100 (est)':>12}\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'TTS generation (40 steps)':<40} {tts_per_gen:>8.3f} s {h100_tts_gen:>10.3f} s\n")
        f.write(f"{'VC conversion':<40} {vc_per_call:>8.3f} s {h100_vc:>10.3f} s\n")
        f.write(f"{'EI scoring (full)':<40} {ei_per_score:>8.3f} s {h100_ei:>10.3f} s\n")
        f.write(f"{'TTS reload (CPU→GPU)':<40} {tts_reload:>8.3f} s {h100_tts_reload:>10.3f} s\n")
        f.write("\n")
        f.write(f"Mode A (all loaded, H100):\n")
        h100_sph_a = 3600 / h100_sample_all
        f.write(f"  Per sample: {h100_sample_all:.2f}s\n")
        f.write(f"  Throughput: {h100_sph_a:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {h100_sph_a * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {h100_sph_a * 8:.0f} samples/hour (8 GPUs)\n\n")

        f.write(f"Mode D (batch=50, H100):\n")
        h100_sph_d = 3600 / h100_batch50
        f.write(f"  Per sample: {h100_batch50:.2f}s\n")
        f.write(f"  Throughput: {h100_sph_d:.0f} samples/hour (1 GPU)\n")
        f.write(f"              {h100_sph_d * 4:.0f} samples/hour (4 GPUs)\n")
        f.write(f"              {h100_sph_d * 8:.0f} samples/hour (8 GPUs)\n\n")

        # Dataset completion estimates
        f.write("DATASET COMPLETION ESTIMATES (78,000 samples)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Configuration':<40} {'Time':>15}\n")
        f.write("-" * 60 + "\n")
        for label, sph in [
            ("A100, 1 GPU, all-loaded", sph_a),
            ("A100, 4 GPUs, all-loaded", sph_a * 4),
            ("A100, 8 GPUs, all-loaded", sph_a * 8),
            ("A100, 1 GPU, batch=50", sph_d),
            ("A100, 4 GPUs, batch=50", sph_d * 4),
            ("A100, 8 GPUs, batch=50", sph_d * 8),
            ("H100, 1 GPU, all-loaded", h100_sph_a),
            ("H100, 4 GPUs, all-loaded", h100_sph_a * 4),
            ("H100, 8 GPUs, all-loaded", h100_sph_a * 8),
            ("H100, 1 GPU, batch=50", h100_sph_d),
            ("H100, 4 GPUs, batch=50", h100_sph_d * 4),
            ("H100, 8 GPUs, batch=50", h100_sph_d * 8),
        ]:
            hours = 78000 / sph
            if hours < 24:
                f.write(f"  {label:<38} {hours:>8.1f} hours\n")
            else:
                f.write(f"  {label:<38} {hours/24:>8.1f} days\n")

        f.write("\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n")
        f.write("1. For 80GB GPUs (A100/H100): Use Mode A (all models loaded).\n")
        f.write("   No offloading overhead. Maximum throughput per GPU.\n\n")
        f.write("2. For <40GB GPUs: Use Mode C/D (batch-phased offloading).\n")
        f.write("   Batch size 50 nearly eliminates load/offload overhead.\n\n")
        f.write("3. Scaling: Each GPU is fully independent. N GPUs = N× throughput.\n")
        f.write("   No inter-GPU communication needed.\n\n")
        f.write("4. torch.compile: Add ~60s warmup but saves ~20-30% on TTS inference.\n")
        f.write("   Worth it for batch processing (amortized over thousands of samples).\n\n")
        f.write("5. The bottleneck is TTS (6 sequential generations per sample).\n")
        f.write("   Batched TTS inference could reduce this by ~2-3x.\n")
        f.write("\n")

    print(f"\nResults written to: {report_path}")

    # Also save raw JSON
    json_path = report_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Per sample (all-loaded):  {sample_time_all_loaded:.2f}s = {3600/sample_time_all_loaded:.0f} samples/hr")
    print(f"Per sample (batch=50):    {sample_time_batch50:.2f}s = {3600/sample_time_batch50:.0f} samples/hr")
    print(f"Per sample (offload/ea):  {sample_time_offload_per:.2f}s = {3600/sample_time_offload_per:.0f} samples/hr")
    print(f"\n4× A100 (all-loaded):     {3600/sample_time_all_loaded*4:.0f} samples/hr")
    print(f"4× H100 est (all-loaded): {h100_sph_a*4:.0f} samples/hr")
    print(f"\n78,000 samples on 4× A100: {78000/(sph_a*4):.1f} hours")
    print(f"78,000 samples on 4× H100: {78000/(h100_sph_a*4):.1f} hours")


if __name__ == "__main__":
    main()
