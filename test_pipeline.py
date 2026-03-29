#!/usr/bin/env python3
"""
End-to-end smoke test for the Voice Acting Pipeline.

Generates a small number of samples and produces an HTML report with
base64-encoded audio players and detailed metadata for quality checking.

Usage:
  python test_pipeline.py --gpu 5 --samples 2 --dimension Anger --bucket 3to4
  python test_pipeline.py --gpu 5  # defaults: 1 sample, first available bucket
"""

import argparse
import base64
import html as html_module
import json
import os
import random
import string
import sys
import time
import traceback

# Ensure we can import from the pipeline directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALL_DIMENSIONS, ECHO_TTS_STEPS, SEEDS_PER_SAMPLE,
    WORD_COUNT_MIN, WORD_COUNT_MAX,
    echo_tts_port, vc_port, ei_port, VLLM_PORT,
    bucket_to_str, get_emotion_description,
)
from dataset_loader import (
    get_emotion_samples, decode_sample_to_wav,
    get_random_laion_voice, load_wav, save_wav, resample_audio,
    download_laion_voices,
)
from sentence_generator import (
    generate_sentence, sample_punctuation_params, get_random_topic,
)


import requests


# Port overrides (set by CLI args for split-GPU setups)
ECHO_PORT_OVERRIDE = None
VC_PORT_OVERRIDE = None
EI_PORT_OVERRIDE = None


def call_echo_tts(text, ref_audio_path, seed, gpu, num_steps=ECHO_TTS_STEPS):
    port = ECHO_PORT_OVERRIDE or echo_tts_port(gpu)
    resp = requests.post(f"http://localhost:{port}/generate",
        data={"text": text, "ref_audio_path": ref_audio_path, "seed": seed, "num_steps": num_steps},
        timeout=300)
    resp.raise_for_status()
    return resp.json()


def call_vc(source_path, target_path, gpu):
    port = VC_PORT_OVERRIDE or vc_port(gpu)
    resp = requests.post(f"http://localhost:{port}/convert",
        data={"source_path": source_path, "target_path": target_path},
        timeout=300)
    resp.raise_for_status()
    return resp.json()


def call_ei(audio_path, gpu):
    port = EI_PORT_OVERRIDE or ei_port(gpu)
    resp = requests.post(f"http://localhost:{port}/score",
        data={"audio_path": audio_path}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def audio_to_base64(wav_path):
    """Convert WAV file to base64 data URI."""
    with open(wav_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def make_audio_player(wav_path, label=""):
    """Create HTML audio player with base64 data."""
    if not os.path.exists(wav_path):
        return f"<em>File not found: {wav_path}</em>"
    data_uri = audio_to_base64(wav_path)
    return (
        f'<div style="margin:4px 0">'
        f'<strong>{html_module.escape(label)}</strong><br>'
        f'<audio controls preload="none" src="{data_uri}"></audio>'
        f'</div>'
    )


def scores_to_html_table(scores, dimension=None):
    """Format EI scores as an HTML table, highlighting the target dimension."""
    if not scores:
        return "<em>No scores available</em>"

    rows = []
    for key, val in sorted(scores.items(), key=lambda x: -abs(x[1])):
        highlight = ' style="background:#ffe0b2;font-weight:bold"' if key == dimension else ""
        rows.append(f"<tr{highlight}><td>{html_module.escape(key)}</td><td>{val:.3f}</td></tr>")

    return (
        '<table style="border-collapse:collapse;font-size:12px">'
        '<tr><th style="text-align:left">Dimension</th><th>Score</th></tr>'
        + "\n".join(rows)
        + "</table>"
    )


def generate_html_report(all_samples, dimension, bucket, output_path):
    """Generate a detailed HTML report with audio players."""
    bucket_str = bucket_to_str(bucket)
    emotion_desc = get_emotion_description(dimension, bucket)

    sections = []
    sections.append(f"""
    <h1>Voice Acting Pipeline - Test Report</h1>
    <p><strong>Dimension:</strong> {html_module.escape(dimension)}</p>
    <p><strong>Bucket:</strong> [{bucket[0]}, {bucket[1]})</p>
    <p><strong>Description:</strong> {html_module.escape(emotion_desc)}</p>
    <p><strong>Samples:</strong> {len(all_samples)}</p>
    <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
    """)

    for si, sample in enumerate(all_samples):
        meta = sample.get("metadata", {})
        emo = meta.get("emotional_sentence", {})
        neu = meta.get("neutral_sentence", {})
        vc_info = meta.get("voice_conversion", {})

        sections.append(f"""
        <h2>Sample {si+1}: {html_module.escape(sample.get('sample_id', ''))}</h2>

        <div style="background:#f5f5f5;padding:10px;border-radius:8px;margin:10px 0">
        <h3>Setup</h3>
        <table>
            <tr><td><strong>Topic:</strong></td><td>{html_module.escape(emo.get('topic', ''))}</td></tr>
            <tr><td><strong>Voice Conversion:</strong></td><td>{'Yes - ' + html_module.escape(vc_info.get('laion_voice', '')) if vc_info.get('used_vc') else 'No (original voice)'}</td></tr>
        </table>
        """)

        # Reference audio
        if sample.get("ref_audio_path") and os.path.exists(sample["ref_audio_path"]):
            sections.append(make_audio_player(sample["ref_audio_path"], "Speaker Reference Audio"))

        sections.append("</div>")

        # Emotional sentence section
        sections.append(f"""
        <div style="background:#fff3e0;padding:10px;border-radius:8px;margin:10px 0">
        <h3>Emotional Sentence ({html_module.escape(dimension)} [{bucket[0]}-{bucket[1]}])</h3>
        <table>
            <tr><td><strong>Text:</strong></td><td style="font-size:16px">{html_module.escape(emo.get('text', ''))}</td></tr>
            <tr><td><strong>Starting Letter:</strong></td><td>{html_module.escape(emo.get('letter', ''))}</td></tr>
            <tr><td><strong>Word Count:</strong></td><td>Target: {emo.get('word_count_target', '')}, Actual: {emo.get('word_count_actual', '')}</td></tr>
            <tr><td><strong>Punctuation:</strong></td><td>{json.dumps(emo.get('punctuation_params', {}))}</td></tr>
            <tr><td><strong>Valid:</strong></td><td>{'Yes' if emo.get('valid') else 'No'}</td></tr>
            <tr><td><strong>Attempts:</strong></td><td>{emo.get('attempts', '')}</td></tr>
        </table>
        """)

        # Emotional audio players + scores
        for gen in meta.get("emotional_generations", []):
            path = gen.get("path", "")
            seed = gen.get("seed", "?")
            duration = gen.get("duration", 0)
            chars_sec = gen.get("chars_per_sec", 0)
            caption = gen.get("caption", "")
            ei_elapsed = gen.get("ei_elapsed", 0)

            sections.append(f"""
            <div style="border:1px solid #e0e0e0;padding:8px;margin:8px 0;border-radius:4px">
            <strong>Seed: {seed}</strong> | Duration: {duration:.2f}s | Chars/sec: {chars_sec:.1f} | EI time: {ei_elapsed:.1f}s
            <br><strong>Caption:</strong> {html_module.escape(caption)}
            """)

            if os.path.exists(path):
                sections.append(make_audio_player(path, f"Emotional (seed {seed})"))

            # Scores
            sections.append("<details><summary>EI Scores</summary>")
            sections.append(scores_to_html_table(gen.get("ei_scores", {}), dimension))
            sections.append("</details></div>")

        sections.append("</div>")

        # Neutral sentence section
        sections.append(f"""
        <div style="background:#e8f5e9;padding:10px;border-radius:8px;margin:10px 0">
        <h3>Neutral/Boring Sentence</h3>
        <table>
            <tr><td><strong>Text:</strong></td><td style="font-size:16px">{html_module.escape(neu.get('text', ''))}</td></tr>
            <tr><td><strong>Starting Letter:</strong></td><td>{html_module.escape(neu.get('letter', ''))}</td></tr>
            <tr><td><strong>Word Count:</strong></td><td>Target: {neu.get('word_count_target', '')}, Actual: {neu.get('word_count_actual', '')}</td></tr>
            <tr><td><strong>Valid:</strong></td><td>{'Yes' if neu.get('valid') else 'No'}</td></tr>
            <tr><td><strong>Attempts:</strong></td><td>{neu.get('attempts', '')}</td></tr>
        </table>
        """)

        for gen in meta.get("neutral_generations", []):
            path = gen.get("path", "")
            seed = gen.get("seed", "?")
            duration = gen.get("duration", 0)
            chars_sec = gen.get("chars_per_sec", 0)
            caption = gen.get("caption", "")
            ei_elapsed = gen.get("ei_elapsed", 0)

            sections.append(f"""
            <div style="border:1px solid #e0e0e0;padding:8px;margin:8px 0;border-radius:4px">
            <strong>Seed: {seed}</strong> | Duration: {duration:.2f}s | Chars/sec: {chars_sec:.1f} | EI time: {ei_elapsed:.1f}s
            <br><strong>Caption:</strong> {html_module.escape(caption)}
            """)

            if os.path.exists(path):
                sections.append(make_audio_player(path, f"Neutral (seed {seed})"))

            sections.append("<details><summary>EI Scores</summary>")
            sections.append(scores_to_html_table(gen.get("ei_scores", {}), dimension))
            sections.append("</details></div>")

        sections.append("</div>")

        # Full metadata JSON
        sections.append("""
        <details><summary>Full Metadata JSON</summary>
        <pre style="background:#f5f5f5;padding:10px;overflow-x:auto;font-size:11px">
        """)
        sections.append(html_module.escape(json.dumps(meta, indent=2, default=str)))
        sections.append("</pre></details><hr>")

    # Assemble HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Voice Acting Pipeline - Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; }}
        td, th {{ padding: 4px 8px; text-align: left; }}
        audio {{ height: 32px; }}
        details {{ margin: 4px 0; }}
        h2 {{ border-top: 2px solid #333; padding-top: 16px; }}
    </style>
</head>
<body>
{"".join(sections)}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"\nHTML report saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")


def run_test(gpu, dimension, bucket, num_samples, output_html):
    """Run end-to-end test with HTML report generation."""
    from worker import process_sample
    from dataset_loader import get_emotion_samples

    bucket_str = bucket_to_str(bucket)
    print(f"\n{'='*60}")
    print(f"TEST: {dimension} [{bucket_str}] on GPU {gpu}")
    print(f"  Samples: {num_samples}")
    print(f"{'='*60}\n")

    # Check server health
    print("Checking server health...", flush=True)
    ttp = ECHO_PORT_OVERRIDE or echo_tts_port(gpu)
    vcp = VC_PORT_OVERRIDE or vc_port(gpu)
    eip = EI_PORT_OVERRIDE or ei_port(gpu)
    servers = [
        (f"http://localhost:{ttp}/health", f"Echo TTS (port {ttp})"),
        (f"http://localhost:{vcp}/health", f"VC (port {vcp})"),
        (f"http://localhost:{eip}/health", f"EI (port {eip})"),
    ]
    for url, name in servers:
        try:
            resp = requests.get(url, timeout=5)
            status = "OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
        except Exception as e:
            status = f"FAILED ({e})"
        print(f"  {name}: {status}")

    # Load emotion references
    print(f"\nLoading emotion references for {dimension} [{bucket_str}]...", flush=True)
    emotion_refs = get_emotion_samples(dimension, bucket)
    if not emotion_refs:
        print(f"ERROR: No samples found for {dimension} [{bucket_str}]")
        return

    print(f"  Found {len(emotion_refs)} reference samples")

    # Process samples
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "test")
    os.makedirs(work_dir, exist_ok=True)

    all_samples = []
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---", flush=True)
        t0 = time.time()
        try:
            sample = process_sample(i, emotion_refs, dimension, bucket, gpu, work_dir)
            all_samples.append(sample)
            elapsed = time.time() - t0
            n_emo = len(sample["emotional_wavs"])
            n_neu = len(sample["neutral_wavs"])
            print(f"  Done in {elapsed:.1f}s ({n_emo} emotional + {n_neu} neutral)")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # Generate HTML report
    if all_samples:
        generate_html_report(all_samples, dimension, bucket, output_html)
    else:
        print("No samples generated - cannot create report")


def main():
    parser = argparse.ArgumentParser(description="Voice Acting Pipeline Smoke Test")
    parser.add_argument("--gpu", type=int, default=5, help="GPU to use")
    parser.add_argument("--dimension", type=str, default="Anger",
                        help="Dimension to test")
    parser.add_argument("--bucket", type=str, default="3to4",
                        help="Bucket to test (e.g. '3to4', 'neg2toneg1')")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path")
    parser.add_argument("--echo-port", type=int, default=None,
                        help="Override Echo TTS port")
    parser.add_argument("--vc-port", type=int, default=None,
                        help="Override VC port")
    parser.add_argument("--ei-port", type=int, default=None,
                        help="Override EI port")
    args = parser.parse_args()

    # Set port overrides
    global ECHO_PORT_OVERRIDE, VC_PORT_OVERRIDE, EI_PORT_OVERRIDE
    if args.echo_port:
        ECHO_PORT_OVERRIDE = args.echo_port
    if args.vc_port:
        VC_PORT_OVERRIDE = args.vc_port
    if args.ei_port:
        EI_PORT_OVERRIDE = args.ei_port

    # Also set in worker module
    import worker
    worker.ECHO_PORT_OVERRIDE = ECHO_PORT_OVERRIDE
    worker.VC_PORT_OVERRIDE = VC_PORT_OVERRIDE
    worker.EI_PORT_OVERRIDE = EI_PORT_OVERRIDE

    # Parse bucket
    bucket_str = args.bucket.replace("neg", "-")
    parts = bucket_str.split("to")
    bucket = (int(parts[0]), int(parts[1]))

    # Default output path
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"test_report_{args.dimension}_{args.bucket}.html"
        )

    run_test(args.gpu, args.dimension, bucket, args.samples, args.output)


if __name__ == "__main__":
    main()
