#!/usr/bin/env python3
"""
Emotion Demo: Generate emotional speech with Christoph & Juniper voices.

For each of 5 emotions × 5 samples × 2 speakers = 50 TTS generations.
Ranks by target emotion score, speech quality, and content enjoyment.
Produces a comprehensive HTML report with compute estimates and pipeline explanation.

Usage:
  LD_LIBRARY_PATH="" python demo_emotions.py
"""

import base64
import html as html_module
import json
import os
import random
import string
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from config import (
    ECHO_TTS_STEPS, WORD_COUNT_MIN, WORD_COUNT_MAX,
    echo_tts_port, vc_port, ei_port,
    bucket_to_str, get_emotion_description, ALL_DIMENSIONS,
)
from dataset_loader import (
    get_emotion_samples, decode_sample_to_wav,
    load_wav, save_wav, resample_audio,
)
from sentence_generator import (
    generate_sentence, sample_punctuation_params, get_random_topic,
)

# ─── Configuration ────────────────────────────────────────────────────────────

EMOTIONS = ["Fear", "Anger", "Sadness", "Amusement", "Longing"]
BUCKET = (3, 4)  # "strongly to extremely present"
SAMPLES_PER_EMOTION = 5
SEEDS = [42, 137, 256, 512, 999]  # Fixed seeds for reproducibility

# Speaker references
CHRISTOPH_REF = os.environ.get("SPEAKER_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ID-refs", "speaker_ref.mp3"))
JUNIPER_REF = os.environ.get("JUNIPER_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ID-refs", "juniper-long-en.wav"))

SPEAKERS = {
    "Christoph": CHRISTOPH_REF,
    "Juniper": JUNIPER_REF,
}

# Server ports
ECHO_PORT = 9205  # GPU 5
EI_PORT = 9403    # GPU 3
VC_PORT = 9302    # GPU 2

# Benchmark data (from benchmark_results.json)
BENCHMARK = {
    "a100_per_sample_40step": 31.43,
    "h100_per_sample_40step": 12.71,
    "a100_tts_per_gen": 4.416,
    "h100_tts_per_gen": 1.766,
    "a100_8step_per_sample": 12.73,
    "h100_8step_per_sample": 5.28,
}

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "demo_emotions")
OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_emotions.html")


# ─── Server Calls ─────────────────────────────────────────────────────────────

def call_echo_tts(text, ref_audio_path, seed, num_steps=ECHO_TTS_STEPS):
    resp = requests.post(f"http://localhost:{ECHO_PORT}/generate",
        data={"text": text, "ref_audio_path": ref_audio_path, "seed": seed, "num_steps": num_steps},
        timeout=300)
    resp.raise_for_status()
    return resp.json()


def call_ei(audio_path):
    resp = requests.post(f"http://localhost:{EI_PORT}/score",
        data={"audio_path": audio_path}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_vc(source_path, target_path):
    """Voice-convert source audio to sound like target speaker."""
    resp = requests.post(f"http://localhost:{VC_PORT}/convert",
        data={"source_path": source_path, "target_path": target_path},
        timeout=120)
    resp.raise_for_status()
    return resp.json()


# ─── HTML Helpers ──────────────────────────────────────────────────────────────

def audio_to_base64(path):
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}.get(ext, "audio/wav")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def audio_player(path, label=""):
    if not path or not os.path.exists(path):
        return f"<em>File not found: {path}</em>"
    uri = audio_to_base64(path)
    return (f'<div style="margin:4px 0"><strong>{html_module.escape(label)}</strong><br>'
            f'<audio controls preload="none" src="{uri}"></audio></div>')


def score_bar(name, value, max_val=4.0, highlight=False):
    pct = min(max(value / max_val * 100, 0), 100)
    color = "#ff9800" if highlight else ("#4caf50" if value > 2.5 else "#2196f3" if value > 1.5 else "#9e9e9e")
    bg = "#fff3e0" if highlight else "#f5f5f5"
    return (f'<div style="display:flex;align-items:center;margin:2px 0;background:{bg};'
            f'padding:2px 6px;border-radius:4px">'
            f'<span style="width:200px;font-size:12px;{"font-weight:bold" if highlight else ""}">'
            f'{html_module.escape(name)}</span>'
            f'<div style="flex:1;height:14px;background:#e0e0e0;border-radius:3px;margin:0 8px">'
            f'<div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:3px"></div></div>'
            f'<span style="font-size:12px;width:50px;text-align:right;{"font-weight:bold" if highlight else ""}">'
            f'{value:.2f}</span></div>')


def ranking_table(items, score_key, title, emotion_name=None):
    """Build a ranking table sorted by score_key."""
    sorted_items = sorted(items, key=lambda x: x.get(score_key, 0), reverse=True)
    rows = []
    for rank, item in enumerate(sorted_items, 1):
        medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
        score = item.get(score_key, 0)
        label = item.get("label", "")
        rows.append(f'<tr><td style="text-align:center;font-size:18px">{medal}</td>'
                     f'<td>{html_module.escape(label)}</td>'
                     f'<td style="text-align:right;font-weight:bold">{score:.3f}</td></tr>')
    return (f'<div style="margin:8px 0"><strong>{html_module.escape(title)}</strong>'
            f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
            f'<tr style="background:#e0e0e0"><th>Rank</th><th>Sample</th><th>Score</th></tr>'
            f'{"".join(rows)}</table></div>')


# ─── Compute Estimates ─────────────────────────────────────────────────────────

def compute_estimates_html():
    """Generate compute estimates section for 10,000 hours/language, 24 languages."""
    samples_per_hour = 120
    hours_per_lang = 10_000
    n_languages = 24
    samples_per_lang = hours_per_lang * samples_per_hour  # 1.2M
    total_samples = samples_per_lang * n_languages  # 28.8M

    b = BENCHMARK

    sections = []
    sections.append(f"""
    <div style="background:linear-gradient(135deg,#1a237e,#283593);color:white;padding:30px;border-radius:12px;margin:20px 0">
    <h2 style="color:white;margin-top:0">Compute Requirements: Full-Scale Dataset Generation</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0">
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;text-align:center">
            <div style="font-size:32px;font-weight:bold">{n_languages}</div>
            <div style="font-size:13px;opacity:0.8">Languages</div>
        </div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;text-align:center">
            <div style="font-size:32px;font-weight:bold">{hours_per_lang:,}</div>
            <div style="font-size:13px;opacity:0.8">Hours per Language</div>
        </div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;text-align:center">
            <div style="font-size:32px;font-weight:bold">{total_samples/1e6:.1f}M</div>
            <div style="font-size:13px;opacity:0.8">Total Sample Groups</div>
        </div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;text-align:center">
            <div style="font-size:32px;font-weight:bold">{hours_per_lang * n_languages:,}</div>
            <div style="font-size:13px;opacity:0.8">Total Audio Hours</div>
        </div>
    </div>
    </div>
    """)

    # Table: GPU configurations
    configs = []
    for gpu_name, per_sample_40, per_sample_8 in [
        ("A100-SXM4-80GB", b["a100_per_sample_40step"], b["a100_8step_per_sample"]),
        ("H100-SXM5-80GB", b["h100_per_sample_40step"], b["h100_8step_per_sample"]),
    ]:
        for steps, per_sample in [("40-step", per_sample_40), ("8-step (consistency)", per_sample_8)]:
            sph = 3600 / per_sample
            for n_gpus in [8, 32, 64, 128, 256]:
                total_sph = sph * n_gpus
                total_hours = total_samples / total_sph
                total_gpu_hours = total_samples * per_sample / 3600
                configs.append({
                    "gpu": gpu_name, "steps": steps, "n_gpus": n_gpus,
                    "per_sample": per_sample, "sph": total_sph,
                    "wall_days": total_hours / 24,
                    "gpu_hours": total_gpu_hours,
                })

    sections.append("""
    <h3>Throughput Comparison</h3>
    <p style="font-size:13px;color:#666">Based on measured A100-SXM4-80GB benchmarks. H100 estimated at 2.5&times; compute speedup.
    Each sample group = 1 VC + 6 TTS generations (40 steps) + 6 EI scorings.</p>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
    <tr style="background:#1a237e;color:white">
        <th style="padding:6px">GPU</th><th>TTS Steps</th><th>GPUs</th>
        <th>Per Sample</th><th>Samples/hr</th><th>Wall Time</th>
        <th>GPU-Hours</th>
    </tr>
    """)

    for i, c in enumerate(configs):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        highlight = c["steps"] == "8-step (consistency)" and c["n_gpus"] == 64
        if highlight:
            bg = "#e8f5e9"
        wall = f'{c["wall_days"]:.1f} days' if c["wall_days"] >= 1 else f'{c["wall_days"]*24:.1f} hrs'
        sections.append(
            f'<tr style="background:{bg};{"font-weight:bold" if highlight else ""}">'
            f'<td style="padding:4px 6px">{c["gpu"][:4]}</td>'
            f'<td>{c["steps"]}</td><td style="text-align:center">{c["n_gpus"]}</td>'
            f'<td style="text-align:right">{c["per_sample"]:.1f}s</td>'
            f'<td style="text-align:right">{c["sph"]:,.0f}</td>'
            f'<td style="text-align:right">{wall}</td>'
            f'<td style="text-align:right">{c["gpu_hours"]:,.0f}</td></tr>')

    sections.append("</table>")

    # Highlight best options
    best_a100 = [c for c in configs if "A100" in c["gpu"] and "8-step" in c["steps"] and c["n_gpus"] == 64][0]
    best_h100 = [c for c in configs if "H100" in c["gpu"] and "8-step" in c["steps"] and c["n_gpus"] == 64][0]

    sections.append(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:20px 0">
    <div style="background:#e8f5e9;padding:15px;border-radius:8px;border:2px solid #4caf50">
        <h4 style="margin-top:0;color:#2e7d32">Best Value: 64&times; A100 + Self-Consistency</h4>
        <ul style="margin:0;font-size:13px">
            <li><strong>{best_a100["wall_days"]:.0f} days</strong> wall-clock time</li>
            <li><strong>{best_a100["gpu_hours"]:,.0f} GPU-hours</strong> total</li>
            <li>240,000 hours of emotion-annotated audio</li>
        </ul>
    </div>
    <div style="background:#e3f2fd;padding:15px;border-radius:8px;border:2px solid #2196f3">
        <h4 style="margin-top:0;color:#1565c0">Fastest: 64&times; H100 + Self-Consistency</h4>
        <ul style="margin:0;font-size:13px">
            <li><strong>{best_h100["wall_days"]:.0f} days</strong> wall-clock time</li>
            <li><strong>{best_h100["gpu_hours"]:,.0f} GPU-hours</strong> total</li>
            <li>240,000 hours of emotion-annotated audio</li>
        </ul>
    </div>
    </div>

    <details style="margin:10px 0">
    <summary style="cursor:pointer;font-weight:bold">Self-Consistency Model Details</summary>
    <div style="padding:10px;background:#f5f5f5;border-radius:8px;font-size:13px">
    <p><strong>What:</strong> Distill the 40-step Euler ODE solver into an 8-step model via consistency distillation.
    The student model has the same architecture but learns to match 40-step quality in only 8 steps.</p>
    <p><strong>Training:</strong> 10&ndash;20 GPU-hours (one-time)</p>
    <p><strong>Quality:</strong> ~95&ndash;98% of 40-step quality. The pipeline's built-in Empathic Insight scoring
    automatically catches any regressions&mdash;low-quality generations are filtered out by cosine similarity selection.</p>
    <p><strong>Speedup:</strong> TTS generation drops from 4.4s to ~1.3s per generation (3.4&times;).
    Since TTS is 84% of compute time, total per-sample time drops from 31.4s to 12.7s (2.5&times;).</p>
    </div>
    </details>
    """)

    return "\n".join(sections)


# ─── Pipeline Explanation ──────────────────────────────────────────────────────

def pipeline_explanation_html():
    """Generate detailed pipeline explanation for newcomers."""
    return """
    <div style="background:#f5f5f5;padding:25px;border-radius:12px;margin:20px 0">
    <h2>How the Voice-Acting Pipeline Works</h2>

    <p style="font-size:14px;line-height:1.6">
    This pipeline generates synthetic training data for a <strong>zero-shot voice and emotion cloning</strong> model.
    The key insight is <strong>identity-emotion disentanglement</strong>: we create audio triplets where speaker identity
    and emotional prosody are explicitly separated, allowing the model to learn them independently.
    </p>

    <h3>The Three Core Models</h3>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:15px 0">

    <div style="background:white;padding:15px;border-radius:8px;border-left:4px solid #ff9800">
        <h4 style="margin-top:0;color:#e65100">Echo TTS (Text-to-Speech)</h4>
        <ul style="font-size:12px;margin:0">
            <li><strong>Architecture:</strong> DiT (Diffusion Transformer) in bf16 + DAC-VAE decoder in fp32</li>
            <li><strong>Size:</strong> ~2.7B parameters, ~7.3 GB VRAM</li>
            <li><strong>Function:</strong> Generates speech from text, cloning the voice and speaking style from a reference audio clip</li>
            <li><strong>Process:</strong> 40-step Euler ODE sampling with CFG (classifier-free guidance) for text and speaker conditioning</li>
            <li><strong>Output:</strong> 44.1 kHz WAV audio</li>
            <li><strong>Speed:</strong> ~4.4s per generation on A100</li>
        </ul>
    </div>

    <div style="background:white;padding:15px;border-radius:8px;border-left:4px solid #2196f3">
        <h4 style="margin-top:0;color:#0d47a1">ChatterboxVC (Voice Conversion)</h4>
        <ul style="font-size:12px;margin:0">
            <li><strong>Architecture:</strong> S3Gen (speech-to-speech generative model) in fp32</li>
            <li><strong>Size:</strong> ~1 GB VRAM</li>
            <li><strong>Function:</strong> Converts the voice identity of audio to match a target speaker while preserving prosody and emotion</li>
            <li><strong>Use case:</strong> Swaps speaker identity to create emotion references with different voices</li>
            <li><strong>Output:</strong> 24 kHz WAV audio</li>
            <li><strong>Speed:</strong> ~1.4s per conversion on A100</li>
            <li><strong>Note:</strong> Runs in a separate Python environment (Python 3.13 via spiritvenv)</li>
        </ul>
    </div>

    <div style="background:white;padding:15px;border-radius:8px;border-left:4px solid #4caf50">
        <h4 style="margin-top:0;color:#1b5e20">Empathic Insight Voice+ (Emotion Scoring)</h4>
        <ul style="font-size:12px;margin:0">
            <li><strong>Architecture:</strong> Whisper encoder (fp16) + 59 specialized MLP expert heads</li>
            <li><strong>Size:</strong> ~8.5 GB VRAM (55 emotion + 4 quality experts)</li>
            <li><strong>Function:</strong> Scores audio across 55 emotion dimensions, 15 voice attributes, and 4 quality metrics</li>
            <li><strong>Emotions:</strong> Fear, Anger, Sadness, Amusement, Longing, Joy, Disgust, Surprise, and 47 more</li>
            <li><strong>Quality metrics:</strong> Overall quality, speech quality, content enjoyment, background quality</li>
            <li><strong>Speed:</strong> ~0.6s per scoring (encoder + all 59 experts + caption) on A100</li>
        </ul>
    </div>
    </div>

    <h3>Pipeline Steps (Per Sample Group)</h3>
    <div style="background:white;padding:15px;border-radius:8px;margin:15px 0">
    <ol style="font-size:13px;line-height:1.8">
        <li><strong>Select Emotion Reference:</strong> Pick a highly emotional audio clip from the
            <a href="https://huggingface.co/datasets/TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave">emotion reference dataset</a>
            (40 emotion dimensions, scored 0&ndash;4, filtered for top bucket 3&ndash;4).</li>
        <li><strong>Voice Conversion (Identity Swap):</strong> 90% of the time, voice-convert the emotion reference
            to a random LAION speaker identity using ChatterboxVC. This creates an emotion reference with a different
            voice, teaching the model to separate emotion from identity.</li>
        <li><strong>Generate Emotional Text:</strong> Use Gemini API to generate an emotional sentence matching
            the target emotion. Includes punctuation control (!, ?, ...) and word count constraints (10&ndash;70 words).
            Simultaneously generate a neutral/boring version of the same topic.</li>
        <li><strong>TTS Generation:</strong> Generate 5 emotional + 1 neutral audio clips using Echo TTS with
            the emotion reference as the speaker/style reference. Each uses a different random seed for diversity.</li>
        <li><strong>Emotion Scoring:</strong> Run all generated clips through Empathic Insight Voice+ to get
            55-dimension emotion vectors. Compare to the original target via cosine similarity.</li>
        <li><strong>Quality Filtering:</strong> Select the generation with highest emotion match.
            Score speech quality and content enjoyment to ensure output quality.</li>
        <li><strong>Package:</strong> Bundle all audio (target, speaker ref, emotion ref, concatenated sequence)
            with DAC-VAE representations and metadata into WebDataset format (.tar shards).</li>
    </ol>
    </div>

    <h3>What This Demo Shows</h3>
    <p style="font-size:13px">
    Below, we generate emotional speech for 5 emotions using two distinct speaker voices (Christoph and Juniper).
    For each emotion, we pick 5 reference audio clips from the dataset and synthesize emotional text using Echo TTS
    with each speaker's voice as the reference. Every generation is scored by Empathic Insight and ranked by:
    </p>
    <ul style="font-size:13px">
        <li><strong>Target Emotion Score:</strong> How well the generated audio matches the intended emotion (0&ndash;4 scale)</li>
        <li><strong>Speech Quality:</strong> Overall speech naturalness and clarity (0&ndash;4 scale)</li>
        <li><strong>Content Enjoyment:</strong> How engaging/enjoyable the speech is to listen to (0&ndash;4 scale)</li>
    </ul>

    <h3>GPU Architecture</h3>
    <p style="font-size:13px">
    All three models fit simultaneously on a single 80GB GPU (~17 GB total), enabling independent parallel workers
    with zero inter-GPU communication. For smaller GPUs (16&ndash;24 GB), models can be loaded/offloaded sequentially
    with batch-phased processing (load model &rarr; process N samples &rarr; offload) at only ~3% throughput penalty
    with batch size 50.
    </p>
    </div>
    """


# ─── Main Generation ──────────────────────────────────────────────────────────

def prepare_speaker_ref(speaker_path, work_dir, max_duration=15.0):
    """Prepare speaker reference: convert to 44.1kHz WAV, trim if needed."""
    os.makedirs(work_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(speaker_path))[0][:30]
    out_path = os.path.join(work_dir, f"{basename}_44k.wav")

    if os.path.exists(out_path):
        return out_path

    # Load (handles MP3 and WAV via soundfile/librosa)
    import torch
    try:
        audio, sr = load_wav(speaker_path)
    except Exception:
        import librosa
        audio_np, sr = librosa.load(speaker_path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np).unsqueeze(0)

    # Convert to mono if stereo
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # Resample to 44.1kHz
    from config import ECHO_TTS_SR
    if sr != ECHO_TTS_SR:
        audio = resample_audio(audio, sr, ECHO_TTS_SR)

    # Trim to max_duration
    max_samples = int(max_duration * ECHO_TTS_SR)
    if audio.shape[-1] > max_samples:
        audio = audio[..., :max_samples]

    save_wav(out_path, audio, ECHO_TTS_SR)
    return out_path


def generate_all():
    """Generate all samples and return structured results."""
    os.makedirs(WORK_DIR, exist_ok=True)

    # Prepare speaker references
    print("Preparing speaker references...", flush=True)
    speaker_refs = {}
    for name, path in SPEAKERS.items():
        ref_path = prepare_speaker_ref(path, WORK_DIR)
        speaker_refs[name] = ref_path
        print(f"  {name}: {ref_path}")

    # Check server health
    print("\nChecking servers...", flush=True)
    for label, url in [
        ("Echo TTS", f"http://localhost:{ECHO_PORT}/health"),
        ("EI", f"http://localhost:{EI_PORT}/health"),
        ("VC", f"http://localhost:{VC_PORT}/health"),
    ]:
        try:
            r = requests.get(url, timeout=5)
            print(f"  {label}: {'OK' if r.status_code == 200 else r.status_code}")
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    all_results = {}
    total_gens = len(EMOTIONS) * SAMPLES_PER_EMOTION * len(SPEAKERS)
    gen_count = 0
    start_time = time.time()

    for emotion in EMOTIONS:
        print(f"\n{'='*60}")
        print(f"EMOTION: {emotion} (bucket {BUCKET})")
        print(f"{'='*60}")

        # Download emotion references
        emotion_samples = get_emotion_samples(emotion, BUCKET)
        if not emotion_samples:
            print(f"  WARNING: No samples found for {emotion} [{BUCKET}], skipping")
            continue

        # Sort by target emotion score (highest first) and pick top N
        def get_emotion_score(s):
            return s.get("json", {}).get(emotion, 0)
        emotion_samples.sort(key=get_emotion_score, reverse=True)
        selected = emotion_samples[:SAMPLES_PER_EMOTION]

        print(f"  Selected {len(selected)} reference samples (out of {len(emotion_samples)})")

        emotion_results = []

        for si, ref_sample in enumerate(selected):
            ref_meta = ref_sample["json"]
            ref_emotion_score = ref_meta.get(emotion, 0)
            ref_caption = ref_meta.get("caption", "")
            sample_id = ref_sample.get("sample_id", f"sample_{si}")

            # Decode emotion reference to WAV
            sample_dir = os.path.join(WORK_DIR, emotion, f"ref_{si}")
            os.makedirs(sample_dir, exist_ok=True)
            ref_wav_path, ref_sr = decode_sample_to_wav(ref_sample, sample_dir)

            print(f"\n  Ref {si+1}/{SAMPLES_PER_EMOTION}: {sample_id} "
                  f"({emotion}={ref_emotion_score:.2f})")

            # Generate emotional sentence
            topic = get_random_topic()
            letter = random.choice(string.ascii_uppercase)
            word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)
            punct = sample_punctuation_params()

            sentence = generate_sentence(
                topic=topic, letter=letter, word_count=word_count,
                dimension=emotion, bucket=BUCKET, punctuation=punct,
                is_emotional=True,
            )
            print(f"    Text: {sentence['text'][:80]}...")

            sample_result = {
                "emotion": emotion,
                "sample_idx": si,
                "sample_id": sample_id,
                "ref_wav_path": ref_wav_path,
                "ref_emotion_score": ref_emotion_score,
                "ref_caption": ref_caption,
                "ref_meta": ref_meta,
                "sentence": sentence,
                "topic": topic,
                "speakers": {},
                "vc_refs": {},
            }

            # Voice-convert emotion reference to each speaker's identity
            # This preserves the emotional prosody while changing the voice
            vc_ref_paths = {}
            for speaker_name, speaker_ref_path in speaker_refs.items():
                vc_out_dir = os.path.join(sample_dir, f"vc_{speaker_name.lower()}")
                os.makedirs(vc_out_dir, exist_ok=True)
                vc_wav_path = os.path.join(vc_out_dir, "vc_emo_ref.wav")

                if not os.path.exists(vc_wav_path):
                    print(f"    VC emotion ref → {speaker_name}...", end=" ", flush=True)
                    try:
                        vc_t0 = time.time()
                        vc_result = call_vc(ref_wav_path, speaker_ref_path)
                        vc_elapsed = time.time() - vc_t0
                        if vc_result.get("status") == "ok":
                            import shutil
                            shutil.copy2(vc_result["output_path"], vc_wav_path)
                            print(f"{vc_elapsed:.1f}s OK")
                        else:
                            print(f"FAILED ({vc_result.get('error', '?')}), using raw ref")
                            vc_wav_path = ref_wav_path
                    except Exception as e:
                        print(f"VC ERROR ({e}), using raw ref")
                        vc_wav_path = ref_wav_path
                else:
                    print(f"    VC {speaker_name}: cached")

                # Resample VC output to 44.1kHz for Echo TTS
                vc_441_path = os.path.join(vc_out_dir, "vc_emo_ref_44k.wav")
                if not os.path.exists(vc_441_path):
                    vc_audio, vc_sr = load_wav(vc_wav_path)
                    from config import ECHO_TTS_SR
                    if vc_sr != ECHO_TTS_SR:
                        vc_audio = resample_audio(vc_audio, vc_sr, ECHO_TTS_SR)
                    # Ensure mono
                    if vc_audio.ndim == 2 and vc_audio.shape[0] > 1:
                        vc_audio = vc_audio.mean(dim=0, keepdim=True)
                    save_wav(vc_441_path, vc_audio, ECHO_TTS_SR)

                vc_ref_paths[speaker_name] = vc_441_path
                sample_result["vc_refs"][speaker_name] = vc_wav_path

            # Generate TTS using VC'd emotion reference as speaker/style ref
            for speaker_name, speaker_ref_path in speaker_refs.items():
                gen_count += 1
                seed = SEEDS[si % len(SEEDS)]
                tts_ref_path = vc_ref_paths[speaker_name]

                print(f"    [{gen_count}/{total_gens}] {speaker_name} (seed={seed})...", end=" ", flush=True)

                try:
                    t0 = time.time()
                    tts_result = call_echo_tts(sentence["text"], tts_ref_path, seed)
                    tts_elapsed = time.time() - t0

                    if tts_result.get("status") != "ok":
                        print(f"TTS FAILED: {tts_result.get('error', 'unknown')}")
                        continue

                    output_path = tts_result["output_path"]
                    duration = tts_result.get("duration", 0)

                    # Score with EI
                    t1 = time.time()
                    ei_result = call_ei(output_path)
                    ei_elapsed = time.time() - t1

                    scores = ei_result.get("scores", {})
                    caption = ei_result.get("caption", "")

                    target_score = scores.get(emotion, 0)
                    speech_quality = scores.get("score_speech_quality", 0)
                    content_enjoyment = scores.get("score_content_enjoyment", 0)
                    overall_quality = scores.get("score_overall_quality", 0)

                    print(f"{tts_elapsed:.1f}s TTS + {ei_elapsed:.1f}s EI | "
                          f"{emotion}={target_score:.2f} quality={speech_quality:.2f}")

                    speaker_result = {
                        "speaker": speaker_name,
                        "seed": seed,
                        "output_path": output_path,
                        "duration": duration,
                        "tts_elapsed": tts_elapsed,
                        "ei_elapsed": ei_elapsed,
                        "caption": caption,
                        "scores": scores,
                        "target_emotion_score": target_score,
                        "speech_quality": speech_quality,
                        "content_enjoyment": content_enjoyment,
                        "overall_quality": overall_quality,
                        "label": f"Ref {si+1} / {speaker_name}",
                    }
                    sample_result["speakers"][speaker_name] = speaker_result

                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()

            emotion_results.append(sample_result)

        all_results[emotion] = emotion_results

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {gen_count} generations in {elapsed:.0f}s ({elapsed/gen_count:.1f}s avg)")
    print(f"{'='*60}")

    return all_results


# ─── HTML Report ───────────────────────────────────────────────────────────────

def build_html(all_results):
    """Build the comprehensive HTML report."""
    sections = []

    # Header
    sections.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Voice-Acting Pipeline: Emotion Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }
        h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
        h2 { color: #283593; margin-top: 40px; }
        h3 { color: #3949ab; }
        audio { height: 36px; width: 100%; }
        table { border-collapse: collapse; }
        td, th { padding: 4px 8px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        details { margin: 6px 0; }
        summary { cursor: pointer; font-weight: bold; color: #1565c0; }
        .emotion-section { border: 2px solid #e0e0e0; border-radius: 12px;
                          padding: 20px; margin: 20px 0; background: white; }
        .sample-card { border: 1px solid #e0e0e0; border-radius: 8px;
                      padding: 15px; margin: 10px 0; background: #fafafa; }
        .speaker-card { border: 1px solid #e0e0e0; border-radius: 8px;
                       padding: 12px; margin: 8px 0; }
        .christoph { background: #fff8e1; border-color: #ffc107; }
        .juniper { background: #e8f5e9; border-color: #4caf50; }
        .ranking-section { background: #f3e5f5; padding: 15px; border-radius: 8px;
                          margin: 15px 0; border: 1px solid #ce93d8; }
    </style>
</head>
<body>
<h1>Voice-Acting Pipeline: Emotion Quality Demo</h1>
<p style="font-size:14px;color:#666">
    Generated: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """ |
    5 emotions &times; 5 samples &times; 2 speakers = 50 TTS generations |
    Ranked by emotion accuracy, speech quality, and content enjoyment
</p>
""")

    # Compute estimates
    sections.append(compute_estimates_html())

    # Pipeline explanation
    sections.append(pipeline_explanation_html())

    # Speaker reference players
    sections.append("""
    <h2>Speaker References</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">
    """)
    for name, path in SPEAKERS.items():
        color = "#fff8e1" if name == "Christoph" else "#e8f5e9"
        sections.append(f"""
        <div style="background:{color};padding:15px;border-radius:8px">
            <h3 style="margin-top:0">{html_module.escape(name)}</h3>
            {audio_player(path, f"{name} Reference Voice")}
        </div>""")
    sections.append("</div>")

    # Results per emotion
    for emotion in EMOTIONS:
        emotion_data = all_results.get(emotion, [])
        if not emotion_data:
            continue

        emotion_desc = get_emotion_description(emotion, BUCKET)

        sections.append(f"""
        <div class="emotion-section">
        <h2 style="margin-top:0">
            {"&#128561;" if emotion == "Fear" else "&#128545;" if emotion == "Anger" else "&#128546;" if emotion == "Sadness" else "&#128514;" if emotion == "Amusement" else "&#128148;"}
            {html_module.escape(emotion)}
            <span style="font-size:14px;font-weight:normal;color:#666">
                (bucket {BUCKET[0]}&ndash;{BUCKET[1]}: {html_module.escape(emotion_desc)})
            </span>
        </h2>
        """)

        # Collect Christoph results for ranking + display
        christoph_items = []
        for sample in emotion_data:
            sr = sample["speakers"].get("Christoph")
            if sr:
                sr["_sample"] = sample  # back-reference for display
                christoph_items.append(sr)

        if christoph_items:
            # Rankings
            sections.append(f"""
            <div class="ranking-section" style="background:#fff8e1;border-color:#ffc107">
            <h3 style="margin-top:0">Christoph &mdash; Rankings</h3>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
            """)
            sections.append(ranking_table(christoph_items, "target_emotion_score",
                                         f"By {emotion} Score", emotion))
            sections.append(ranking_table(christoph_items, "speech_quality",
                                         "By Speech Quality"))
            sections.append(ranking_table(christoph_items, "content_enjoyment",
                                         "By Content Enjoyment"))
            sections.append("</div></div>")

            # All Christoph samples with audio players, ranked by target emotion
            sorted_items = sorted(christoph_items, key=lambda x: x.get("target_emotion_score", 0), reverse=True)
            for rank, sr in enumerate(sorted_items, 1):
                sample = sr["_sample"]
                si = sample["sample_idx"]
                medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
                border_color = '#ff9800' if rank == 1 else '#ffc107' if rank <= 3 else '#e0e0e0'
                ref_wav = sample.get('ref_wav_path', '')
                vc_wav = sample.get('vc_refs', {}).get('Christoph', '')
                ref_emo_score = sample.get('ref_emotion_score', 0)
                text_escaped = html_module.escape(sample['sentence']['text'])
                topic_escaped = html_module.escape(sample.get('topic', ''))
                ref_player = audio_player(ref_wav, f"1. Emotion Reference ({emotion} = {ref_emo_score:.2f})")
                vc_player = audio_player(vc_wav, "2. VC'd to Christoph (emotion prosody + Christoph voice)")
                tts_player = audio_player(sr['output_path'], f"3. Echo TTS Output (seed {sr['seed']})")

                sections.append(f"""
                <div class="sample-card" style="border-left:4px solid {border_color}">
                <h4 style="margin-top:0">{medal} Sample {si+1} &mdash; {emotion} = {sr['target_emotion_score']:.3f}</h4>
                <table style="font-size:13px">
                    <tr><td><strong>Text:</strong></td>
                        <td><em>{text_escaped}</em></td></tr>
                    <tr><td><strong>Topic:</strong></td><td>{topic_escaped}</td></tr>
                </table>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:8px">
                    <div style="background:#f5f5f5;padding:8px;border-radius:6px">
                        {ref_player}
                    </div>
                    <div style="background:#fff8e1;padding:8px;border-radius:6px">
                        {vc_player}
                    </div>
                    <div style="background:#e8f5e9;padding:8px;border-radius:6px">
                        {tts_player}
                    </div>
                </div>
                <table style="font-size:12px;width:100%;margin-top:6px">
                    <tr><td>Duration: {sr['duration']:.2f}s</td>
                        <td><strong>{emotion}: {sr['target_emotion_score']:.3f}</strong></td>
                        <td>Quality: {sr['speech_quality']:.3f}</td>
                        <td>Enjoyment: {sr['content_enjoyment']:.3f}</td>
                        <td>Overall: {sr['overall_quality']:.3f}</td></tr>
                </table>
                <p style="font-size:11px;color:#666;margin:4px 0"><strong>Caption:</strong> {html_module.escape(sr['caption'][:200])}</p>
                """)

                # Score bars
                sections.append('<details><summary style="font-size:12px">All EI Scores</summary>')
                emotion_scores = {k: v for k, v in sr["scores"].items()
                                 if k in set(["Fear", "Anger", "Sadness", "Amusement", "Longing",
                                             "Elation", "Contentment", "Disgust", "Distress",
                                             "Pain", "Helplessness", "Hope_Enthusiasm_Optimism",
                                             "Bitterness", "Contempt", "Doubt",
                                             "score_speech_quality", "score_content_enjoyment",
                                             "score_overall_quality", "score_background_quality",
                                             "Arousal", "Valence"])}
                for key in sorted(emotion_scores, key=lambda k: emotion_scores[k], reverse=True):
                    val = emotion_scores[key]
                    is_target = key == emotion
                    sections.append(score_bar(key, val, highlight=is_target))
                sections.append('</details></div>')

        sections.append('</div>')  # emotion-section

    # Footer
    sections.append("""
    <div style="margin-top:40px;padding:20px;background:#e0e0e0;border-radius:8px;text-align:center;font-size:12px;color:#666">
        <p>Generated by the Voice-Acting Pipeline | Models: Echo TTS + ChatterboxVC + Empathic Insight Voice+</p>
        <p>GPU: NVIDIA A100-SXM4-80GB | Benchmark data from single-GPU throughput measurements</p>
    </div>
    </body></html>""")

    html_content = "\n".join(sections)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)

    # Clean up back-references added during rendering
    for emotion_data in all_results.values():
        if isinstance(emotion_data, list):
            for sample in emotion_data:
                if isinstance(sample, dict):
                    for spk_data in sample.get("speakers", {}).values():
                        if isinstance(spk_data, dict):
                            spk_data.pop("_sample", None)

    size_mb = os.path.getsize(OUTPUT_HTML) / 1024 / 1024
    print(f"\nHTML report saved: {OUTPUT_HTML} ({size_mb:.1f} MB)")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild HTML from existing JSON (no TTS generation)")
    args = parser.parse_args()

    if args.rebuild:
        print("Rebuilding HTML from existing JSON...")
        json_path = OUTPUT_HTML.replace(".html", ".json")
        with open(json_path) as f:
            results = json.load(f)
        build_html(results)
    else:
        print("=" * 60)
        print("EMOTION DEMO: 5 emotions × 5 samples × 2 speakers")
        print("=" * 60)

        results = generate_all()
        build_html(results)

    # Save raw JSON for later analysis
    json_path = OUTPUT_HTML.replace(".html", ".json")
    # Strip non-serializable items
    clean = {}
    for emo, samples in results.items():
        clean[emo] = []
        for s in samples:
            cs = {k: v for k, v in s.items() if k != "ref_meta"}
            cs["sentence"] = s["sentence"]
            cs["speakers"] = {}
            for spk, sr in s["speakers"].items():
                cs["speakers"][spk] = {k: v for k, v in sr.items()}
            clean[emo].append(cs)
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"Raw data saved: {json_path}")
