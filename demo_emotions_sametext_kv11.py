#!/usr/bin/env python3
"""
Emotion Demo (Same Text) with Speaker KV Scaling = 1.1

Same setup as demo_emotions_sametext.py but uses speaker_kv_scale=1.1
to gently boost speaker/emotion conditioning in attention layers.
Reuses the same sentences from the baseline run for fair comparison.

Usage:
  LD_LIBRARY_PATH="" python demo_emotions_sametext_kv11.py
"""

import base64
import html as html_module
import json
import os
import random
import shutil
import string
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from config import (
    ECHO_TTS_STEPS, WORD_COUNT_MIN, WORD_COUNT_MAX, ECHO_TTS_SR,
    get_emotion_description,
)
from dataset_loader import (
    get_emotion_samples, decode_sample_to_wav,
    load_wav, save_wav, resample_audio,
)
from sentence_generator import get_random_topic

# ─── Configuration ────────────────────────────────────────────────────────────

EMOTIONS = ["Fear", "Anger", "Sadness", "Amusement", "Longing"]
BUCKET = (3, 4)
SAMPLES_PER_EMOTION = 5
SEEDS = [42, 137, 256, 512, 999, 1337, 2024, 3141, 4269, 7777]  # 10 seeds per ref

CHRISTOPH_REF = os.environ.get("SPEAKER_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ID-refs", "speaker_ref.mp3"))

ECHO_PORT = 9205
EI_PORT = 9403
VC_PORT = 9302

SPEAKER_KV_SCALE = 1.1

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tmp", "demo_emotions_sametext_kv11")
OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "demo_emotions_sametext_kv11.html")

# Baseline JSON for reusing sentences
BASELINE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "demo_emotions_sametext.json")


# ─── Sentence Generation ─────────────────────────────────────────────────────

def generate_sentence_expressive(topic, letter, word_count, dimension, bucket):
    """Generate an expressive sentence via the LLM."""
    from sentence_generator import query_llm, validate_sentence, VLLM_PORT

    emotion_desc = get_emotion_description(dimension, bucket)

    system = ("You are a passionate, emotionally expressive voice actor script writer. "
              "Output ONLY the sentence, nothing else. No quotes, no labels, no explanation.")

    user = (
        f"Write a single emotionally intense sentence about: {topic}\n\n"
        f"Requirements:\n"
        f"- The sentence MUST begin with the capital letter '{letter}'\n"
        f"- Approximately {word_count} words long\n"
        f"- Express extreme '{dimension}' at a level described as: {emotion_desc}\n"
        f"- Use normal sentence casing only — do NOT capitalize words for emphasis\n"
        f"- Use expressive punctuation naturally: !, ?, ..., ?!, !!\n"
        f"- Use emotional interjections where natural (Oh!, God!, Ugh!, Gosh!, Please!)\n"
        f"- Make it sound like someone speaking with intense genuine emotion\n"
        f"- Output ONLY the sentence, nothing else"
    )

    best = None
    for _ in range(3):
        try:
            raw = query_llm(system, user, port=VLLM_PORT)
            valid, issues, cleaned = validate_sentence(raw, letter, word_count)
            if valid or best is None:
                best = cleaned
            if valid:
                break
        except Exception:
            pass

    return best or f"{letter}nknown sentence generation failed."


# ─── Server Calls ────────────────────────────────────────────────────────────

def call_echo_tts(text, ref_audio_path, seed, num_steps=ECHO_TTS_STEPS,
                  speaker_kv_scale=SPEAKER_KV_SCALE):
    data = {
        "text": text,
        "ref_audio_path": ref_audio_path,
        "seed": seed,
        "num_steps": num_steps,
        "speaker_kv_scale": speaker_kv_scale,
    }
    r = requests.post(f"http://localhost:{ECHO_PORT}/generate",
        data=data, timeout=300)
    r.raise_for_status()
    return r.json()


def call_ei(audio_path):
    r = requests.post(f"http://localhost:{EI_PORT}/score",
        data={"audio_path": audio_path}, timeout=120)
    r.raise_for_status()
    return r.json()


def call_vc(source_path, target_path):
    r = requests.post(f"http://localhost:{VC_PORT}/convert",
        data={"source_path": source_path, "target_path": target_path}, timeout=120)
    r.raise_for_status()
    return r.json()


# ─── HTML Helpers ─────────────────────────────────────────────────────────────

def audio_to_base64(path):
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = {".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(ext, "audio/wav")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


def audio_player(path, label=""):
    if not path or not os.path.exists(path):
        return f"<em>File not found</em>"
    uri = audio_to_base64(path)
    return (f'<div style="margin:4px 0"><strong>{html_module.escape(label)}</strong><br>'
            f'<audio controls preload="none" src="{uri}"></audio></div>')


def score_bar(name, value, max_val=4.0, highlight=False):
    pct = min(max(value / max_val * 100, 0), 100)
    color = "#4caf50" if highlight else ("#4caf50" if value > 2.5 else "#2196f3" if value > 1.5 else "#9e9e9e")
    bg = "#e8f5e9" if highlight else "#f5f5f5"
    return (f'<div style="display:flex;align-items:center;margin:2px 0;background:{bg};'
            f'padding:2px 6px;border-radius:4px">'
            f'<span style="width:200px;font-size:12px;{"font-weight:bold" if highlight else ""}">'
            f'{html_module.escape(name)}</span>'
            f'<div style="flex:1;height:14px;background:#e0e0e0;border-radius:3px;margin:0 8px">'
            f'<div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:3px"></div></div>'
            f'<span style="font-size:12px;width:50px;text-align:right;{"font-weight:bold" if highlight else ""}">'
            f'{value:.2f}</span></div>')


# ─── Speaker Reference Prep ──────────────────────────────────────────────────

def prepare_speaker_ref(speaker_path, work_dir, max_duration=15.0):
    os.makedirs(work_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(speaker_path))[0][:30]
    out_path = os.path.join(work_dir, f"{basename}_44k.wav")
    if os.path.exists(out_path):
        return out_path

    import torch
    try:
        audio, sr = load_wav(speaker_path)
    except Exception:
        import librosa
        audio_np, sr = librosa.load(speaker_path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np).unsqueeze(0)

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif audio.ndim == 1:
        audio = audio.unsqueeze(0)

    if sr != ECHO_TTS_SR:
        audio = resample_audio(audio, sr, ECHO_TTS_SR)

    max_samples = int(max_duration * ECHO_TTS_SR)
    if audio.shape[-1] > max_samples:
        audio = audio[..., :max_samples]

    save_wav(out_path, audio, ECHO_TTS_SR)
    return out_path


# ─── Main Generation ─────────────────────────────────────────────────────────

def generate_all():
    os.makedirs(WORK_DIR, exist_ok=True)

    christoph_ref = prepare_speaker_ref(CHRISTOPH_REF, WORK_DIR)
    print(f"Speaker: Christoph → {christoph_ref}")
    print(f"Speaker KV Scale: {SPEAKER_KV_SCALE}")

    # Load baseline sentences for reuse
    baseline_data = {}
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON) as f:
            baseline_data = json.load(f)
        print(f"Loaded baseline sentences from {BASELINE_JSON}")
    else:
        print("WARNING: No baseline JSON found, generating new sentences")

    # Health check
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
    n_seeds = len(SEEDS)
    total_gens = len(EMOTIONS) * SAMPLES_PER_EMOTION * n_seeds
    gen_count = 0
    start_time = time.time()

    for emotion in EMOTIONS:
        print(f"\n{'='*60}")
        print(f"EMOTION: {emotion} (bucket {BUCKET})")
        print(f"{'='*60}")

        # Get emotion references
        emotion_samples = get_emotion_samples(emotion, BUCKET)
        if not emotion_samples:
            print(f"  WARNING: No samples for {emotion}, skipping")
            continue

        emotion_samples.sort(
            key=lambda s: s.get("json", {}).get(emotion, 0), reverse=True)
        selected = emotion_samples[:SAMPLES_PER_EMOTION]
        print(f"  Selected {len(selected)} refs (out of {len(emotion_samples)})")

        # Reuse sentence from baseline, or generate new one
        baseline_emo = baseline_data.get(emotion, {})
        if baseline_emo and "sentence" in baseline_emo:
            sentence_text = baseline_emo["sentence"]
            topic = baseline_emo.get("topic", "unknown")
            print(f"\n  Reusing baseline sentence: {sentence_text[:100]}...")
        else:
            topic = get_random_topic()
            letter = random.choice(string.ascii_uppercase)
            word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)
            sentence_text = generate_sentence_expressive(
                topic, letter, word_count, emotion, BUCKET)
            print(f"\n  Generated new sentence: {sentence_text[:100]}...")
        print(f"  Topic: {topic} | Seeds: {n_seeds} per ref | KV scale: {SPEAKER_KV_SCALE}")

        emotion_results = {
            "sentence": sentence_text,
            "topic": topic,
            "seeds": SEEDS,
            "samples": [],
        }

        for si, ref_sample in enumerate(selected):
            ref_meta = ref_sample["json"]
            ref_emotion_score = ref_meta.get(emotion, 0)
            ref_caption = ref_meta.get("caption", "")
            sample_id = ref_sample.get("sample_id", f"sample_{si}")

            # Decode emotion reference
            sample_dir = os.path.join(WORK_DIR, emotion, f"ref_{si}")
            os.makedirs(sample_dir, exist_ok=True)
            ref_wav_path, ref_sr = decode_sample_to_wav(ref_sample, sample_dir)

            print(f"\n  Ref {si+1}/{SAMPLES_PER_EMOTION}: {sample_id} "
                  f"({emotion}={ref_emotion_score:.2f})")

            # VC emotion ref → Christoph identity
            vc_dir = os.path.join(sample_dir, "vc_christoph")
            os.makedirs(vc_dir, exist_ok=True)
            vc_wav_path = os.path.join(vc_dir, "vc_emo_ref.wav")

            if not os.path.exists(vc_wav_path):
                print(f"    VC → Christoph...", end=" ", flush=True)
                try:
                    t0 = time.time()
                    vc_result = call_vc(ref_wav_path, christoph_ref)
                    vc_elapsed = time.time() - t0
                    if vc_result.get("status") == "ok":
                        shutil.copy2(vc_result["output_path"], vc_wav_path)
                        print(f"{vc_elapsed:.1f}s OK")
                    else:
                        print(f"FAILED, using raw ref")
                        vc_wav_path = ref_wav_path
                except Exception as e:
                    print(f"VC ERROR ({e}), using raw ref")
                    vc_wav_path = ref_wav_path
            else:
                print(f"    VC: cached")

            # Resample VC output to 44.1kHz
            vc_441_path = os.path.join(vc_dir, "vc_emo_ref_44k.wav")
            if not os.path.exists(vc_441_path):
                vc_audio, vc_sr = load_wav(vc_wav_path)
                if vc_sr != ECHO_TTS_SR:
                    vc_audio = resample_audio(vc_audio, vc_sr, ECHO_TTS_SR)
                if vc_audio.ndim == 2 and vc_audio.shape[0] > 1:
                    vc_audio = vc_audio.mean(dim=0, keepdim=True)
                save_wav(vc_441_path, vc_audio, ECHO_TTS_SR)

            # TTS with 10 seeds, keep best by target emotion score
            seed_results = []
            for seed in SEEDS:
                gen_count += 1
                print(f"    [{gen_count}/{total_gens}] seed={seed}...", end=" ", flush=True)

                try:
                    t0 = time.time()
                    tts_result = call_echo_tts(sentence_text, vc_441_path, seed)
                    tts_elapsed = time.time() - t0

                    if tts_result.get("status") != "ok":
                        print(f"FAIL")
                        continue

                    output_path = tts_result["output_path"]
                    duration = tts_result.get("duration", 0)

                    t1 = time.time()
                    ei_result = call_ei(output_path)
                    ei_elapsed = time.time() - t1

                    scores = ei_result.get("scores", {})
                    caption = ei_result.get("caption", "")

                    target_score = scores.get(emotion, 0)
                    speech_quality = scores.get("score_speech_quality", 0)
                    content_enjoyment = scores.get("score_content_enjoyment", 0)
                    overall_quality = scores.get("score_overall_quality", 0)

                    print(f"{tts_elapsed:.1f}s+{ei_elapsed:.1f}s | "
                          f"{emotion}={target_score:.2f} q={speech_quality:.2f}")

                    seed_results.append({
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
                    })
                except Exception as e:
                    print(f"ERROR: {e}")

            if not seed_results:
                continue

            # Pick best seed by target emotion score
            best = max(seed_results, key=lambda x: x["target_emotion_score"])
            avg_emo = sum(r["target_emotion_score"] for r in seed_results) / len(seed_results)
            max_emo = best["target_emotion_score"]
            min_emo = min(r["target_emotion_score"] for r in seed_results)

            print(f"    → Best seed={best['seed']}: {emotion}={max_emo:.2f} "
                  f"(avg={avg_emo:.2f}, min={min_emo:.2f})")

            emotion_results["samples"].append({
                "ref_idx": si,
                "sample_id": sample_id,
                "ref_wav_path": ref_wav_path,
                "ref_emotion_score": ref_emotion_score,
                "ref_caption": ref_caption,
                "vc_wav_path": vc_wav_path,
                # Best seed result
                "best_seed": best["seed"],
                "output_path": best["output_path"],
                "duration": best["duration"],
                "caption": best["caption"],
                "scores": best["scores"],
                "target_emotion_score": best["target_emotion_score"],
                "speech_quality": best["speech_quality"],
                "content_enjoyment": best["content_enjoyment"],
                "overall_quality": best["overall_quality"],
                # Stats across all seeds
                "avg_emotion": avg_emo,
                "min_emotion": min_emo,
                "max_emotion": max_emo,
                "all_seeds": seed_results,
                "label": f"Ref {si+1} ({emotion}={ref_emotion_score:.2f})",
            })

        all_results[emotion] = emotion_results

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {gen_count} generations in {elapsed:.0f}s ({elapsed/gen_count:.1f}s avg)")
    print(f"{'='*60}")

    return all_results


# ─── HTML Report ──────────────────────────────────────────────────────────────

EI_SHOW_KEYS = {
    "Fear", "Anger", "Sadness", "Amusement", "Longing",
    "Elation", "Contentment", "Disgust", "Distress",
    "Pain", "Helplessness", "Hope_Enthusiasm_Optimism",
    "Bitterness", "Contempt", "Doubt",
    "score_speech_quality", "score_content_enjoyment",
    "score_overall_quality", "score_background_quality",
    "Arousal", "Valence",
}


def build_html(all_results):
    h = []
    n_seeds = len(SEEDS)

    h.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Emotion Demo: Same Text, KV Scale 1.1</title>
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
    .seed-row { display: flex; align-items: center; gap: 8px; padding: 3px 0;
                font-size: 12px; border-bottom: 1px solid #eee; }
    .seed-row audio { height: 30px; flex: 1; }
</style>
</head>
<body>
<h1>Emotion Demo: Same Text, Speaker KV Scale = 1.1</h1>
""")

    h.append(f"""<p style="font-size:14px;color:#666">
    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} |
    5 emotions &times; 5 emotion refs &times; {n_seeds} seeds = {5*5*n_seeds} TTS generations |
    <strong>speaker_kv_scale = {SPEAKER_KV_SCALE}</strong> |
    Best of {n_seeds} shown, all {n_seeds} playable
</p>""")

    # Explanation box
    h.append(f"""
    <div style="background:#e3f2fd;padding:20px;border-radius:12px;margin:20px 0;border:2px solid #2196f3">
    <h3 style="margin-top:0;color:#0d47a1">KV Scaling Experiment: speaker_kv_scale = {SPEAKER_KV_SCALE}</h3>
    <p style="font-size:14px;line-height:1.6;margin-bottom:0">
    This is a comparison run against the <strong>baseline</strong> (default settings, no KV scaling)
    and the <strong>KV 1.5 experiment</strong>. Everything is identical &mdash; same sentences,
    same emotion refs, same seeds &mdash; except that <code>speaker_kv_scale={SPEAKER_KV_SCALE}</code>
    is applied, which gently boosts the attention weights for speaker/emotion conditioning by {int(SPEAKER_KV_SCALE*100-100)}%.
    </p>
    <p style="font-size:14px;line-height:1.6;margin-bottom:0">
    KV 1.5 was too aggressive: it doubled Amusement scores but destroyed Fear and Longing.
    KV 1.1 tests whether a subtle 10% boost can improve emotion transfer across all emotions
    without the destructive overcorrection seen at 1.5.
    </p>
    <p style="font-size:13px;line-height:1.6;margin-bottom:0;color:#555">
    <strong>Pipeline:</strong>
    Emotion Ref (scored 4.0) &rarr;
    ChatterboxVC (to Christoph) &rarr;
    Echo TTS ({n_seeds} seeds, <strong>kv_scale={SPEAKER_KV_SCALE}</strong>) &rarr;
    Empathic Insight (score each) &rarr;
    Pick best seed
    </p>
    </div>
    """)

    # Speaker ref player
    h.append(f"""
    <div style="background:#fff8e1;padding:15px;border-radius:8px;margin:15px 0;display:inline-block">
        <h3 style="margin-top:0">Christoph (Speaker Reference)</h3>
        {audio_player(CHRISTOPH_REF, "Christoph Reference Voice")}
    </div>
    """)

    for emotion in EMOTIONS:
        edata = all_results.get(emotion)
        if not edata:
            continue

        samples = edata.get("samples", [])
        if not samples:
            continue

        emotion_desc = get_emotion_description(emotion, BUCKET)
        emoji = {"Fear": "&#128561;", "Anger": "&#128545;", "Sadness": "&#128546;",
                 "Amusement": "&#128514;", "Longing": "&#128148;"}.get(emotion, "")

        h.append(f"""
        <div class="emotion-section">
        <h2 style="margin-top:0">
            {emoji} {html_module.escape(emotion)}
            <span style="font-size:14px;font-weight:normal;color:#666">
                (bucket {BUCKET[0]}&ndash;{BUCKET[1]}: {html_module.escape(emotion_desc)})
            </span>
        </h2>
        """)

        # Shared sentence box
        h.append(f"""
        <div style="background:#e8eaf6;padding:15px;border-radius:8px;margin:10px 0;
                    border-left:4px solid #3f51b5">
            <div style="font-size:12px;color:#5c6bc0;font-weight:bold;margin-bottom:4px">
                SHARED SENTENCE (all 5 refs use this exact text)</div>
            <div style="font-size:16px;font-style:italic;line-height:1.5">
                &ldquo;{html_module.escape(edata['sentence'])}&rdquo;
            </div>
            <div style="font-size:12px;color:#666;margin-top:6px">
                Topic: {html_module.escape(edata['topic'])} |
                {n_seeds} seeds per ref, best shown
            </div>
        </div>
        """)

        # Ranking table (by best-of-N emotion score)
        sorted_samples = sorted(samples,
            key=lambda x: x.get("target_emotion_score", 0), reverse=True)

        h.append(f"""
        <div style="background:#fff8e1;padding:15px;border-radius:8px;margin:15px 0;
                    border:1px solid #ffc107">
        <h3 style="margin-top:0">Rankings (best of {n_seeds} seeds per ref)</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px">
        """)

        # By emotion (best seed)
        h.append(f'<div><strong>By {emotion} (best seed)</strong>')
        h.append('<table style="width:100%;font-size:13px">')
        for rank, s in enumerate(sorted_samples, 1):
            medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
            avg = s.get("avg_emotion", 0)
            h.append(f'<tr><td style="text-align:center">{medal}</td>'
                     f'<td>{html_module.escape(s["label"])}</td>'
                     f'<td style="text-align:right;font-weight:bold">'
                     f'{s["target_emotion_score"]:.3f}</td>'
                     f'<td style="text-align:right;font-size:11px;color:#888">'
                     f'avg {avg:.2f}</td></tr>')
        h.append('</table></div>')

        # By quality
        by_quality = sorted(samples,
            key=lambda x: x.get("speech_quality", 0), reverse=True)
        h.append('<div><strong>By Speech Quality</strong>')
        h.append('<table style="width:100%;font-size:13px">')
        for rank, s in enumerate(by_quality, 1):
            medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
            h.append(f'<tr><td style="text-align:center">{medal}</td>'
                     f'<td>{html_module.escape(s["label"])}</td>'
                     f'<td style="text-align:right;font-weight:bold">'
                     f'{s["speech_quality"]:.3f}</td></tr>')
        h.append('</table></div>')

        # By enjoyment
        by_enjoy = sorted(samples,
            key=lambda x: x.get("content_enjoyment", 0), reverse=True)
        h.append('<div><strong>By Content Enjoyment</strong>')
        h.append('<table style="width:100%;font-size:13px">')
        for rank, s in enumerate(by_enjoy, 1):
            medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
            h.append(f'<tr><td style="text-align:center">{medal}</td>'
                     f'<td>{html_module.escape(s["label"])}</td>'
                     f'<td style="text-align:right;font-weight:bold">'
                     f'{s["content_enjoyment"]:.3f}</td></tr>')
        h.append('</table></div>')
        h.append('</div></div>')  # close grid + ranking box

        # Individual sample cards, ranked by target emotion
        for rank, s in enumerate(sorted_samples, 1):
            medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(rank, f"#{rank}")
            border_color = '#2196f3' if rank == 1 else '#64b5f6' if rank <= 3 else '#e0e0e0'

            ref_player = audio_player(s['ref_wav_path'],
                f"1. Original Emotion Ref ({emotion}={s['ref_emotion_score']:.2f})")
            vc_player = audio_player(s['vc_wav_path'],
                "2. VC'd to Christoph")
            tts_player = audio_player(s['output_path'],
                f"3. Best TTS (seed {s['best_seed']})")

            avg_emo = s.get('avg_emotion', 0)
            min_emo = s.get('min_emotion', 0)
            max_emo = s.get('max_emotion', 0)

            h.append(f"""
            <div class="sample-card" style="border-left:4px solid {border_color}">
            <h4 style="margin-top:0">
                {medal} Ref {s['ref_idx']+1} &mdash;
                best {emotion} = {s['target_emotion_score']:.3f}
                <span style="font-size:12px;font-weight:normal;color:#666">
                    (avg={avg_emo:.2f}, min={min_emo:.2f}, max={max_emo:.2f} |
                     ref={s['ref_emotion_score']:.2f} | seed={s['best_seed']})
                </span>
            </h4>

            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
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
                <tr>
                    <td>Duration: {s['duration']:.2f}s</td>
                    <td><strong>{emotion}: {s['target_emotion_score']:.3f}</strong></td>
                    <td>Quality: {s['speech_quality']:.3f}</td>
                    <td>Enjoyment: {s['content_enjoyment']:.3f}</td>
                    <td>Overall: {s['overall_quality']:.3f}</td>
                </tr>
            </table>
            <p style="font-size:11px;color:#666;margin:4px 0">
                <strong>Caption:</strong> {html_module.escape(s['caption'][:200])}
            </p>
            """)

            # All seed variations as a collapsible list with audio players
            all_seeds = s.get("all_seeds", [])
            if all_seeds:
                sorted_seeds = sorted(all_seeds,
                    key=lambda x: x.get("target_emotion_score", 0), reverse=True)
                h.append(f'<details><summary style="font-size:12px">'
                         f'All {len(all_seeds)} seed variations</summary>'
                         f'<div style="margin:6px 0">')
                for sr in sorted_seeds:
                    emo_val = sr.get("target_emotion_score", 0)
                    q_val = sr.get("speech_quality", 0)
                    is_best = sr["seed"] == s["best_seed"]
                    bg = "#e8f5e9" if is_best else "transparent"
                    bold = "font-weight:bold;" if is_best else ""
                    star = " ★" if is_best else ""
                    out_path = sr.get("output_path", "")
                    if out_path and os.path.exists(out_path):
                        uri = audio_to_base64(out_path)
                        h.append(
                            f'<div class="seed-row" style="background:{bg};{bold}">'
                            f'<span style="width:70px">seed {sr["seed"]}{star}</span>'
                            f'<span style="width:90px">{emotion}={emo_val:.2f}</span>'
                            f'<span style="width:60px">q={q_val:.2f}</span>'
                            f'<audio controls preload="none" src="{uri}"></audio>'
                            f'</div>')
                    else:
                        h.append(
                            f'<div class="seed-row">'
                            f'<span>seed {sr["seed"]}: {emotion}={emo_val:.2f} q={q_val:.2f}</span>'
                            f'</div>')
                h.append('</div></details>')

            # EI score bars for best seed
            h.append('<details><summary style="font-size:12px">All EI Scores (best seed)</summary>')
            emo_scores = {k: v for k, v in s["scores"].items() if k in EI_SHOW_KEYS}
            for key in sorted(emo_scores, key=lambda k: emo_scores[k], reverse=True):
                h.append(score_bar(key, emo_scores[key], highlight=(key == emotion)))
            h.append('</details></div>')

        h.append('</div>')  # emotion-section

    # Footer
    h.append(f"""
    <div style="margin-top:40px;padding:20px;background:#e0e0e0;border-radius:8px;
                text-align:center;font-size:12px;color:#666">
        <p>Generated by the Voice-Acting Pipeline | Models: Echo TTS + ChatterboxVC + Empathic Insight Voice+</p>
        <p>GPU: NVIDIA A100-SXM4-80GB | speaker_kv_scale={SPEAKER_KV_SCALE} | Same text, {n_seeds} seeds per ref</p>
    </div>
    </body></html>""")

    html_content = "\n".join(h)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)

    size_mb = os.path.getsize(OUTPUT_HTML) / 1024 / 1024
    print(f"\nHTML report saved: {OUTPUT_HTML} ({size_mb:.1f} MB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild:
        json_path = OUTPUT_HTML.replace(".html", ".json")
        with open(json_path) as f:
            results = json.load(f)
        build_html(results)
    else:
        print("=" * 60)
        print(f"SAME-TEXT DEMO (KV Scale {SPEAKER_KV_SCALE}): 5 emotions × 5 refs × 1 speaker")
        print("=" * 60)
        results = generate_all()
        build_html(results)

    # Save JSON
    json_path = OUTPUT_HTML.replace(".html", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw data saved: {json_path}")
