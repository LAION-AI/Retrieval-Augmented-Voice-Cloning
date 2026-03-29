#!/usr/bin/env python3
"""
Build HTML demo pages from pipeline output tars.
Creates an index page + per-bucket demo pages with embedded audio players.
"""

import os
import sys
import tarfile
import json
import base64
import io
from collections import defaultdict

TARS_DIR = "/tmp/hf_demo_tars/data"
OUTPUT_DIR = "/tmp/hf_demo_html"

# Score interpretation guides
SCORE_GUIDES = {
    # Emotions (0-4 scale, softmax probability)
    "Anger": {"scale": "0-6", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4+=extreme"},
    "Amusement": {"scale": "0-5", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4+=extreme"},
    "Affection": {"scale": "0-5", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4+=extreme"},
    "Awe": {"scale": "0-5", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4+=extreme"},
    "Astonishment_Surprise": {"scale": "0-5", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4+=extreme"},
    "Concentration": {"scale": "0-4", "desc": "0=none, 1=slight, 2=moderate, 3=strong, 4=extreme"},
    # Attributes with special scales
    "Arousal": {"scale": "0-4", "desc": "0=very calm, 1=calm, 2=neutral, 3=excited, 4=very excited"},
    "Age": {"scale": "0-6", "desc": "0=infant, 1=child, 2=teenager, 3=young adult, 4=adult, 5=senior, 6=very old"},
    "Background_Noise": {"scale": "0-3", "desc": "0=no noise, 1=slight, 2=moderate, 3=intense"},
    "Authenticity": {"scale": "0-4", "desc": "0=artificial, 1=somewhat, 2=decent, 3=natural, 4=genuine"},
    "Valence": {"scale": "-3 to +3", "desc": "-3=extreme negative, 0=neutral, +3=extreme positive"},
    "Submissive_vs._Dominant": {"scale": "-3 to +3", "desc": "-3=very submissive, 0=neutral, +3=very dominant"},
    "Gender": {"scale": "-2 to +2", "desc": "-2=very masculine, 0=neutral, +2=very feminine"},
    "Warm_vs._Cold": {"scale": "-2 to +2", "desc": "-2=very cold, 0=neutral, +2=very warm"},
    "Soft_vs._Harsh": {"scale": "-2 to +2", "desc": "-2=very harsh, 0=neutral, +2=very soft"},
    "Monotone_vs._Expressive": {"scale": "0-4", "desc": "0=very monotone, 2=neutral, 4=very expressive"},
    "High-Pitched_vs._Low-Pitched": {"scale": "0-4", "desc": "0=very low, 2=neutral, 4=very high"},
    "Confident_vs._Hesitant": {"scale": "0-4", "desc": "0=very confident, 2=neutral, 4=very hesitant"},
    "Vulnerable_vs._Emotionally_Detached": {"scale": "0-4", "desc": "0=very vulnerable, 2=neutral, 4=detached"},
    "Serious_vs._Humorous": {"scale": "0-4", "desc": "0=very serious, 2=neutral, 4=very humorous"},
    "Recording_Quality": {"scale": "0-4", "desc": "0=very low, 2=decent, 4=very high"},
}

# Top emotions to highlight per dimension
def get_top_scores(scores, dimension, n=8):
    """Get top N relevant scores for display."""
    # Always include the target dimension first
    sorted_scores = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    result = []
    # Add target dimension first
    if dimension in scores:
        result.append((dimension, scores[dimension]))
    # Add top by magnitude
    for k, v in sorted_scores:
        if k != dimension and len(result) < n:
            result.append((k, v))
    return result


def score_bar_html(name, value, dim_name):
    """Create a colored score bar."""
    guide = SCORE_GUIDES.get(name, {"scale": "0-4", "desc": ""})

    # Determine color based on value
    if value < 0:
        pct = min(abs(value) / 3.0 * 100, 100)
        color = f"hsl({240 - abs(value)*30}, 70%, 50%)"  # blue to purple
        bar_style = f"width:{pct}%;background:{color};margin-left:auto"
    else:
        pct = min(value / 4.0 * 100, 100)
        if name == dim_name:
            color = f"hsl({120 - value*20}, 80%, 45%)"  # green -> yellow -> orange
        else:
            color = f"hsl(210, 60%, {65 - value*8}%)"  # light blue -> dark blue
        bar_style = f"width:{pct}%;background:{color}"

    is_target = " (TARGET)" if name == dim_name else ""
    label = name.replace("_", " ")

    return f'''<div style="display:flex;align-items:center;margin:2px 0;font-size:12px">
        <span style="width:200px;text-align:right;padding-right:8px;color:{'#e74c3c' if name==dim_name else '#666'};font-weight:{'bold' if name==dim_name else 'normal'}">{label}{is_target}</span>
        <div style="width:200px;height:14px;background:#f0f0f0;border-radius:3px;overflow:hidden">
            <div style="{bar_style};height:100%;border-radius:3px"></div>
        </div>
        <span style="width:60px;padding-left:6px;font-weight:bold">{value:.2f}</span>
    </div>'''


def audio_player_html(audio_bytes, label=""):
    """Create base64 audio player."""
    b64 = base64.b64encode(audio_bytes).decode()
    return f'''<div style="margin:4px 0">
        <span style="font-size:12px;color:#666">{label}</span>
        <audio controls style="height:32px;vertical-align:middle">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    </div>'''


def build_bucket_page(tar_path, output_dir):
    """Build HTML page for one bucket tar."""
    tar_name = os.path.basename(tar_path)
    # Parse dimension and bucket from tar name
    # Format: Anger_4to5_7979252679.tar
    parts = tar_name.replace(".tar", "").rsplit("_", 1)[0]  # Remove random digits
    # Find the bucket part (NtoN)
    tokens = parts.split("_")
    bucket_str = tokens[-1]  # e.g. "4to5"
    dimension = "_".join(tokens[:-1])  # e.g. "Anger" or "Astonishment_Surprise"

    page_id = f"{dimension}_{bucket_str}"
    guide = SCORE_GUIDES.get(dimension, {"scale": "?", "desc": "emotion/attribute dimension"})

    samples = []
    audio_data = {}

    with tarfile.open(tar_path) as tf:
        # First pass: read all files
        for member in tf.getmembers():
            if member.name.endswith('.json'):
                f = tf.extractfile(member)
                data = json.load(f)
                samples.append(data)
            elif member.name.endswith('.wav'):
                f = tf.extractfile(member)
                audio_data[member.name] = f.read()

    samples.sort(key=lambda x: x['sample_id'])

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{dimension} [{bucket_str}] - Voice Acting Pipeline Demo</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.sample {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.sample-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
.sample-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
.meta {{ font-size: 13px; color: #7f8c8d; }}
.text-box {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 10px 15px; margin: 8px 0; border-radius: 0 8px 8px 0; }}
.emotional {{ border-left-color: #e74c3c; }}
.neutral {{ border-left-color: #95a5a6; }}
.audio-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 10px 0; }}
.audio-section {{ background: #f8f9fa; border-radius: 8px; padding: 12px; }}
.audio-section h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
.scores-section {{ margin-top: 15px; }}
.vc-badge {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
.orig-badge {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
.caption-box {{ background: #fff3cd; border-radius: 6px; padding: 8px 12px; margin: 4px 0; font-size: 12px; font-style: italic; }}
a {{ color: #3498db; }}
.back {{ margin-bottom: 20px; }}
</style>
</head>
<body>
<div class="back"><a href="index.html">&larr; Back to Index</a></div>
<h1>{dimension.replace("_", " ")} [{bucket_str}]</h1>
<p><strong>Scale:</strong> {guide["scale"]} &mdash; {guide["desc"]}</p>
<p><strong>Bucket range:</strong> [{bucket_str.replace("to", ", ")})</p>
<p><strong>Samples:</strong> {len(samples)} (each with 3 emotional + 3 neutral generations)</p>
'''

    for sample in samples:
        sid = sample['sample_id']
        vc = sample.get('voice_conversion', {})
        vc_used = vc.get('used_vc', False)
        vc_badge = f'<span class="vc-badge">VC: {vc.get("laion_voice","?")}</span>' if vc_used else '<span class="orig-badge">Original Voice</span>'

        emo_sent = sample.get('emotional_sentence', {})
        neu_sent = sample.get('neutral_sentence', {})

        html += f'''
<div class="sample">
    <div class="sample-header">
        <span class="sample-title">{sid}</span>
        <span>{vc_badge}</span>
    </div>
    <div class="meta">
        Topic: <strong>{emo_sent.get("topic", "?")}</strong> |
        VC time: {vc.get("vc_elapsed", 0):.1f}s
    </div>

    <h4 style="color:#e74c3c;margin:12px 0 4px">Emotional Sentence:</h4>
    <div class="text-box emotional">"{emo_sent.get("text", "?")}"</div>
    <div class="meta">Letter: {emo_sent.get("letter","?")} | Target words: {emo_sent.get("word_count_target",0)} | Actual: {emo_sent.get("word_count_actual",0)} | Punctuation: ! x{emo_sent.get("punctuation_params",{}).get("exclamation_count",0)}, ? x{emo_sent.get("punctuation_params",{}).get("question_count",0)}, ... {"yes" if emo_sent.get("punctuation_params",{}).get("use_ellipsis") else "no"}</div>

    <h4 style="color:#95a5a6;margin:12px 0 4px">Neutral Sentence:</h4>
    <div class="text-box neutral">"{neu_sent.get("text", "?")}"</div>
    <div class="meta">Letter: {neu_sent.get("letter","?")} | Target words: {neu_sent.get("word_count_target",0)} | Actual: {neu_sent.get("word_count_actual",0)}</div>

    <h3 style="margin:15px 0 8px">Reference Speaker Audio</h3>
'''
        # Reference audio
        ref_key = f"{sid}.ref_audio.wav"
        if ref_key in audio_data:
            html += audio_player_html(audio_data[ref_key], "Speaker reference (used for TTS conditioning)")

        # Emotional generations
        emo_gens = sample.get('emotional_generations', [])
        html += '<h3 style="margin:15px 0 8px;color:#e74c3c">Emotional Generations (3 seeds)</h3>'
        html += '<div class="audio-grid">'

        for gen in emo_gens:
            seed = gen['seed']
            wav_key = f"{sid}.emotional_seed{seed}.wav"
            elapsed = gen.get('elapsed', 0)
            duration = gen.get('duration', 0)
            ei_scores = gen.get('ei_scores', {})
            caption = gen.get('ei_caption', '')

            # Get target dimension score
            target_score = ei_scores.get(dimension, 0)

            html += f'''<div class="audio-section">
                <h4>Seed {seed} <span style="font-weight:normal;color:#666">({elapsed:.1f}s gen, {duration:.1f}s audio)</span></h4>
                <div style="font-size:13px;margin-bottom:4px"><strong>{dimension.replace("_"," ")} score: <span style="color:#e74c3c">{target_score:.2f}</span></strong></div>'''

            if wav_key in audio_data:
                html += audio_player_html(audio_data[wav_key])

            if caption:
                html += f'<div class="caption-box">{caption}</div>'

            # Top scores
            top = get_top_scores(ei_scores, dimension, 6)
            for name, val in top:
                html += score_bar_html(name, val, dimension)

            html += '</div>'

        html += '</div>'

        # Neutral generations
        neu_gens = sample.get('neutral_generations', [])
        html += '<h3 style="margin:15px 0 8px;color:#95a5a6">Neutral Generations (3 seeds)</h3>'
        html += '<div class="audio-grid">'

        for gen in neu_gens:
            seed = gen['seed']
            wav_key = f"{sid}.neutral_seed{seed}.wav"
            elapsed = gen.get('elapsed', 0)
            duration = gen.get('duration', 0)
            ei_scores = gen.get('ei_scores', {})
            caption = gen.get('ei_caption', '')

            target_score = ei_scores.get(dimension, 0)

            html += f'''<div class="audio-section">
                <h4>Seed {seed} <span style="font-weight:normal;color:#666">({elapsed:.1f}s gen, {duration:.1f}s audio)</span></h4>
                <div style="font-size:13px;margin-bottom:4px"><strong>{dimension.replace("_"," ")} score: <span style="color:#3498db">{target_score:.2f}</span></strong></div>'''

            if wav_key in audio_data:
                html += audio_player_html(audio_data[wav_key])

            if caption:
                html += f'<div class="caption-box">{caption}</div>'

            top = get_top_scores(ei_scores, dimension, 6)
            for name, val in top:
                html += score_bar_html(name, val, dimension)

            html += '</div>'

        html += '</div>'

        # Full EI scores (collapsed)
        if emo_gens and emo_gens[0].get('ei_scores'):
            all_scores = emo_gens[0]['ei_scores']
            html += '''<details style="margin-top:10px">
                <summary style="cursor:pointer;color:#3498db;font-size:13px">Show all 55 EI scores (seed ''' + str(emo_gens[0]['seed']) + ''')</summary>
                <div style="column-count:2;margin-top:8px">'''
            for name in sorted(all_scores.keys()):
                html += score_bar_html(name, all_scores[name], dimension)
            html += '</div></details>'

        html += '</div>'  # end sample

    html += '''
</body>
</html>'''

    out_path = os.path.join(output_dir, f"{page_id}.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"  Built {page_id}.html ({len(samples)} samples)")
    return page_id, dimension, bucket_str, len(samples)


def build_index(pages, output_dir):
    """Build index page linking to all bucket pages."""
    # Group by dimension
    by_dim = defaultdict(list)
    for page_id, dim, bucket, n_samples in pages:
        by_dim[dim].append((page_id, bucket, n_samples))

    # Sort dimensions, emotions first then attributes
    emotions = sorted([d for d in by_dim if d not in [
        "Arousal", "Age", "Background_Noise", "Authenticity", "Valence",
        "Submissive_vs._Dominant", "Gender", "Warm_vs._Cold", "Soft_vs._Harsh",
        "Monotone_vs._Expressive", "High-Pitched_vs._Low-Pitched",
        "Confident_vs._Hesitant", "Vulnerable_vs._Emotionally_Detached",
        "Serious_vs._Humorous", "Recording_Quality", "Concentration"
    ]])
    attributes = sorted([d for d in by_dim if d not in emotions])

    html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Voice Acting Pipeline - Demo Index</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; }
.dim-group { background: white; border-radius: 12px; padding: 15px 20px; margin: 10px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
.dim-name { font-size: 18px; font-weight: bold; color: #2c3e50; }
.bucket-links { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px; }
.bucket-link { display: inline-block; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 500; transition: all 0.2s; }
.bucket-link:hover { transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.15); }
.low { background: #d4edda; color: #155724; }
.mid { background: #fff3cd; color: #856404; }
.high { background: #f8d7da; color: #721c24; }
.guide { font-size: 13px; color: #7f8c8d; margin-top: 4px; }
.stats { background: #e8f4fd; border-radius: 8px; padding: 15px; margin: 20px 0; }
</style>
</head>
<body>
<h1>Voice Acting Pipeline - Demo Samples</h1>
<div class="stats">
    <strong>Pipeline:</strong> Echo TTS + ChatterboxVC + Empathic Insight Voice+ |
    <strong>Total pages:</strong> ''' + str(len(pages)) + ''' |
    <strong>Total samples:</strong> ''' + str(sum(n for _, _, _, n in pages)) + ''' |
    <strong>Total WAVs:</strong> ''' + str(sum(n * 6 for _, _, _, n in pages)) + '''
    <br><strong>Each sample:</strong> 3 emotional seeds + 3 neutral seeds, scored by Empathic Insight (55 dimensions)
    <br><strong>Dataset:</strong> <a href="https://huggingface.co/datasets/TTS-AGI/voice-acting-pipeline-output">TTS-AGI/voice-acting-pipeline-output</a>
</div>
'''

    if emotions:
        html += '<h2>Emotions (40 dimensions, 0-4+ scale)</h2>'
        html += '<p style="color:#666">Each emotion is bucketed by intensity. Higher buckets = stronger emotion in the reference audio used for TTS conditioning.</p>'
        for dim in emotions:
            buckets = sorted(by_dim[dim], key=lambda x: x[1])
            guide = SCORE_GUIDES.get(dim, {"scale": "?", "desc": ""})
            html += f'''<div class="dim-group">
                <div class="dim-name">{dim.replace("_", " ")}</div>
                <div class="guide">{guide["desc"]}</div>
                <div class="bucket-links">'''
            for page_id, bucket, n in buckets:
                # Determine color class
                lo = float(bucket.split("to")[0])
                cls = "low" if lo < 2 else ("mid" if lo < 3 else "high")
                html += f'<a href="{page_id}.html" class="bucket-link {cls}">[{bucket.replace("to","-")}] ({n} samples)</a>'
            html += '</div></div>'

    if attributes:
        html += '<h2>Attributes (15 dimensions, varying scales)</h2>'
        html += '<p style="color:#666">Attributes have dimension-specific scales. Some have meaningful negative/zero values (e.g., Arousal 0 = very calm).</p>'
        for dim in attributes:
            buckets = sorted(by_dim[dim], key=lambda x: x[1])
            guide = SCORE_GUIDES.get(dim, {"scale": "?", "desc": ""})
            html += f'''<div class="dim-group">
                <div class="dim-name">{dim.replace("_", " ")}</div>
                <div class="guide">Scale: {guide["scale"]} &mdash; {guide["desc"]}</div>
                <div class="bucket-links">'''
            for page_id, bucket, n in buckets:
                lo = float(bucket.split("to")[0])
                cls = "low" if lo < 2 else ("mid" if lo < 3 else "high")
                html += f'<a href="{page_id}.html" class="bucket-link {cls}">[{bucket.replace("to","-")}] ({n} samples)</a>'
            html += '</div></div>'

    html += '''
<hr style="margin-top:40px">
<p style="color:#999;font-size:12px">Generated by Voice Acting Pipeline demo builder. Data from <a href="https://huggingface.co/datasets/TTS-AGI/voice-acting-pipeline-output">TTS-AGI/voice-acting-pipeline-output</a></p>
</body>
</html>'''

    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Built index.html with {len(pages)} pages")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tar_files = sorted([
        os.path.join(TARS_DIR, f) for f in os.listdir(TARS_DIR)
        if f.endswith('.tar')
    ])

    print(f"Found {len(tar_files)} tar files to process")

    pages = []
    for tar_path in tar_files:
        try:
            result = build_bucket_page(tar_path, OUTPUT_DIR)
            pages.append(result)
        except Exception as e:
            print(f"  ERROR processing {tar_path}: {e}")
            import traceback
            traceback.print_exc()

    build_index(pages, OUTPUT_DIR)
    print(f"\nDone! HTML files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
