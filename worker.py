"""
Per-GPU worker: orchestrates the generation loop.

OPTIMIZED VERSION:
- Pipelines TTS generation with EI scoring (different GPUs, can overlap)
- Generates emotional + neutral sentences concurrently
- Uses ThreadPoolExecutor for parallel I/O
- Submits EI scoring as soon as each TTS output is ready

For each bucket in a claimed dimension:
1. Download emotion references from HF
2. For each of 10 samples:
   a. Pick random emotion reference from bucket
   b. Roll d10: 10% keep original, 90% voice-convert to LAION ref speaker
   c. Generate emotional + neutral sentences concurrently via Gemini
   d. Pipeline: generate TTS audio, submit EI scoring as each WAV completes
   e. Record metadata
3. Package into WebDataset tar and upload to HF
"""

import json
import os
import random
import shutil
import string
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import (
    GPUS, SAMPLES_PER_BUCKET, SEEDS_PER_SAMPLE, ECHO_TTS_STEPS,
    WORD_COUNT_MIN, WORD_COUNT_MAX, SPEAKER_REF_MIN_DURATION,
    SPEAKER_REF_MAX_DURATION, VLLM_PORT, TMP_DIR, PROGRESS_DIR,
    ALL_DIMENSIONS, ECHO_TTS_SR, CHATTERBOX_SR, DACVAE_SR,
    echo_tts_port, vc_port, ei_port, bucket_to_str,
)
from dataset_loader import (
    get_emotion_samples, decode_sample_to_wav,
    get_random_laion_voice, get_audio_duration,
    resample_audio, load_wav, save_wav,
)
from sentence_generator import (
    generate_sentence, sample_punctuation_params, get_random_topic,
)
from uploader import package_and_upload


# Port overrides for split-GPU setups (set externally)
ECHO_PORT_OVERRIDE = None
VC_PORT_OVERRIDE = None
EI_PORT_OVERRIDE = None

# Shared thread pool for async EI scoring
_ei_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ei")
_sentence_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")


def call_echo_tts(text, ref_audio_path, seed, gpu, num_steps=ECHO_TTS_STEPS):
    """Call Echo TTS server to generate audio."""
    port = ECHO_PORT_OVERRIDE or echo_tts_port(gpu)
    resp = requests.post(
        f"http://localhost:{port}/generate",
        data={
            "text": text,
            "ref_audio_path": ref_audio_path,
            "seed": seed,
            "num_steps": num_steps,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_vc(source_path, target_path, gpu):
    """Call voice conversion server."""
    port = VC_PORT_OVERRIDE or vc_port(gpu)
    resp = requests.post(
        f"http://localhost:{port}/convert",
        data={
            "source_path": source_path,
            "target_path": target_path,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_ei(audio_path, gpu):
    """Call Empathic Insight server for scoring + captioning."""
    port = EI_PORT_OVERRIDE or ei_port(gpu)
    resp = requests.post(
        f"http://localhost:{port}/score",
        data={"audio_path": audio_path},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def is_bucket_done(dimension, bucket):
    """Check if a bucket has already been processed (progress file exists)."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    bucket_str = bucket_to_str(bucket)
    progress_file = os.path.join(PROGRESS_DIR, f"{dimension}_{bucket_str}.done")
    return os.path.exists(progress_file)


def mark_bucket_done(dimension, bucket):
    """Mark a bucket as completed."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    bucket_str = bucket_to_str(bucket)
    progress_file = os.path.join(PROGRESS_DIR, f"{dimension}_{bucket_str}.done")
    with open(progress_file, "w") as f:
        f.write(str(time.time()))


def process_sample(
    sample_idx: int,
    emotion_refs: list,
    dimension: str,
    bucket: tuple,
    gpu: int,
    work_dir: str,
) -> dict:
    """Process a single sample with pipelined TTS+EI.

    Key optimization: EI scoring runs in background threads while TTS
    generates the next WAV (they use different GPUs).
    """
    sample_dir = os.path.join(work_dir, f"sample_{sample_idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Pick random emotion reference
    ref_sample = random.choice(emotion_refs)
    ref_meta = ref_sample["json"]

    # Decode emotion reference to WAV (at DACVAE native rate = 48kHz)
    ref_wav_path, ref_sr = decode_sample_to_wav(ref_sample, sample_dir)

    # 2. Roll d10: 10% keep original, 90% voice-convert to LAION ref speaker
    #    Exception: never VC for pitch/age dimensions (VC destroys these properties)
    from config import NO_VC_DIMENSIONS
    skip_vc = dimension in NO_VC_DIMENSIONS
    use_original_voice = skip_vc or random.random() < 0.1
    speaker_ref_path = ref_wav_path
    vc_target_path = None
    vc_info = {"used_vc": False, "skipped_reason": "pitch/age dimension" if skip_vc else None}

    if not use_original_voice:
        try:
            laion_voice_path = get_random_laion_voice()
            vc_target_path = laion_voice_path
            vc_result = call_vc(ref_wav_path, laion_voice_path, gpu)
            if vc_result.get("status") == "ok":
                speaker_ref_path = vc_result["output_path"]
                vc_info = {
                    "used_vc": True,
                    "laion_voice": os.path.basename(laion_voice_path),
                    "vc_elapsed": vc_result.get("elapsed", 0),
                }
            else:
                print(f"  VC failed, using original: {vc_result.get('error')}")
        except Exception as e:
            print(f"  VC error, using original: {e}")

    # Resample speaker ref to 44.1kHz for Echo TTS
    spk_audio, spk_sr = load_wav(speaker_ref_path)
    if spk_sr != ECHO_TTS_SR:
        spk_audio = resample_audio(spk_audio, spk_sr, ECHO_TTS_SR)
    spk_ref_441_path = os.path.join(sample_dir, "speaker_ref_44k.wav")
    save_wav(spk_ref_441_path, spk_audio, ECHO_TTS_SR)

    # Trim speaker ref to 6-15 seconds
    spk_duration = spk_audio.shape[-1] / ECHO_TTS_SR
    if spk_duration > SPEAKER_REF_MAX_DURATION:
        max_samples = int(SPEAKER_REF_MAX_DURATION * ECHO_TTS_SR)
        spk_audio_trimmed = spk_audio[..., :max_samples]
        save_wav(spk_ref_441_path, spk_audio_trimmed, ECHO_TTS_SR)

    # 3. Generate emotional + neutral sentences CONCURRENTLY
    emo_topic = get_random_topic()
    emo_letter = random.choice(string.ascii_uppercase)
    emo_word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)
    emo_punct = sample_punctuation_params()

    neu_topic = emo_topic
    neu_letter = random.choice([l for l in string.ascii_uppercase if l != emo_letter])
    neu_word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)

    # Submit both sentence generations concurrently
    emo_future = _sentence_pool.submit(
        generate_sentence,
        topic=emo_topic, letter=emo_letter, word_count=emo_word_count,
        dimension=dimension, bucket=bucket, punctuation=emo_punct,
        is_emotional=True,
    )
    neu_future = _sentence_pool.submit(
        generate_sentence,
        topic=neu_topic, letter=neu_letter, word_count=neu_word_count,
        is_emotional=False,
    )

    emo_sentence = emo_future.result()
    neu_sentence = neu_future.result()

    # 4. PIPELINED TTS + EI: generate all 6 WAVs, submit EI scoring immediately
    emotional_wavs = []
    neutral_wavs = []
    emotional_results = []
    neutral_results = []
    ei_futures = []  # (future, gen_dict, label, text) for background EI scoring

    seeds = [random.randint(0, 999999) for _ in range(SEEDS_PER_SAMPLE)]

    for seed in seeds:
        # Emotional TTS
        try:
            emo_result = call_echo_tts(
                emo_sentence["text"], spk_ref_441_path, seed, gpu)
            if emo_result.get("status") == "ok":
                emotional_wavs.append((seed, emo_result["output_path"]))
                gen = {
                    "seed": seed,
                    "path": emo_result["output_path"],
                    "duration": emo_result.get("duration", 0),
                    "elapsed": emo_result.get("elapsed", 0),
                }
                emotional_results.append(gen)
                # Submit EI scoring immediately (runs on different GPU)
                fut = _ei_pool.submit(call_ei, gen["path"], gpu)
                ei_futures.append((fut, gen, "emotional", emo_sentence["text"]))
        except Exception as e:
            print(f"  Echo TTS emotional seed={seed} failed: {e}")

        # Neutral TTS
        try:
            neu_result = call_echo_tts(
                neu_sentence["text"], spk_ref_441_path, seed, gpu)
            if neu_result.get("status") == "ok":
                neutral_wavs.append((seed, neu_result["output_path"]))
                gen = {
                    "seed": seed,
                    "path": neu_result["output_path"],
                    "duration": neu_result.get("duration", 0),
                    "elapsed": neu_result.get("elapsed", 0),
                }
                neutral_results.append(gen)
                # Submit EI scoring immediately
                fut = _ei_pool.submit(call_ei, gen["path"], gpu)
                ei_futures.append((fut, gen, "neutral", neu_sentence["text"]))
        except Exception as e:
            print(f"  Echo TTS neutral seed={seed} failed: {e}")

    # 5. Collect all EI scoring results
    for fut, gen, label, text in ei_futures:
        try:
            ei_result = fut.result(timeout=180)
            if ei_result.get("status") == "ok":
                gen["ei_scores"] = ei_result["scores"]
                gen["caption"] = ei_result.get("caption", "")
                gen["ei_elapsed"] = ei_result.get("elapsed", 0)
            else:
                gen["ei_scores"] = {}
                gen["caption"] = ""
        except Exception as e:
            print(f"  EI scoring failed for {label} seed={gen['seed']}: {e}")
            gen["ei_scores"] = {}
            gen["caption"] = ""

        duration = gen.get("duration", 0)
        gen["chars_per_sec"] = round(len(text) / duration, 2) if duration > 0 else 0

    # 6. Build metadata
    bucket_str = bucket_to_str(bucket)
    sample_id = f"{dimension}_{bucket_str}_{sample_idx:03d}"

    metadata = {
        "sample_id": sample_id,
        "dimension": dimension,
        "bucket": list(bucket),
        "bucket_str": bucket_str,
        "voice_conversion": vc_info,
        "source_ref": {
            "sample_id": ref_sample.get("sample_id", ""),
            "metadata_keys": list(ref_meta.keys())[:5],
        },
        "emotional_sentence": {
            "text": emo_sentence["text"],
            "topic": emo_topic,
            "letter": emo_letter,
            "word_count_target": emo_word_count,
            "word_count_actual": emo_sentence["word_count_actual"],
            "punctuation_params": emo_punct,
            "valid": emo_sentence["valid"],
            "attempts": emo_sentence["attempts"],
        },
        "neutral_sentence": {
            "text": neu_sentence["text"],
            "topic": neu_topic,
            "letter": neu_letter,
            "word_count_target": neu_word_count,
            "word_count_actual": neu_sentence["word_count_actual"],
            "valid": neu_sentence["valid"],
            "attempts": neu_sentence["attempts"],
        },
        "emotional_generations": emotional_results,
        "neutral_generations": neutral_results,
    }

    return {
        "sample_id": sample_id,
        "emotional_wavs": emotional_wavs,
        "neutral_wavs": neutral_wavs,
        "ref_audio_path": spk_ref_441_path,
        "metadata": metadata,
        "work_dir": sample_dir,
    }


def process_bucket(dimension, bucket, gpu, upload=True):
    """Process a full bucket: generate SAMPLES_PER_BUCKET samples and upload."""
    bucket_str = bucket_to_str(bucket)
    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {dimension} [{bucket_str}] on GPU {gpu}", flush=True)
    print(f"{'='*60}", flush=True)

    if is_bucket_done(dimension, bucket):
        print(f"  Skipping (already done)")
        return []

    # Download emotion references for this bucket
    t0 = time.time()
    emotion_refs = get_emotion_samples(dimension, bucket)
    if not emotion_refs:
        print(f"  No samples found for {dimension} [{bucket_str}], skipping")
        return []
    print(f"  Loaded {len(emotion_refs)} emotion references in {time.time()-t0:.1f}s")

    # Create working directory
    work_dir = os.path.join(TMP_DIR, f"gpu{gpu}", f"{dimension}_{bucket_str}")
    os.makedirs(work_dir, exist_ok=True)

    # Process samples
    samples = []
    bucket_start = time.time()
    for i in range(SAMPLES_PER_BUCKET):
        print(f"\n  Sample {i+1}/{SAMPLES_PER_BUCKET}", flush=True)
        t0 = time.time()
        try:
            sample = process_sample(i, emotion_refs, dimension, bucket, gpu, work_dir)
            samples.append(sample)
            elapsed = time.time() - t0
            n_emo = len(sample["emotional_wavs"])
            n_neu = len(sample["neutral_wavs"])
            rate = (i + 1) / (time.time() - bucket_start)
            eta = (SAMPLES_PER_BUCKET - i - 1) / rate if rate > 0 else 0
            print(f"    Done in {elapsed:.1f}s ({n_emo} emo + {n_neu} neu) | ETA: {eta:.0f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    bucket_elapsed = time.time() - bucket_start
    print(f"\n  Bucket complete: {len(samples)}/{SAMPLES_PER_BUCKET} samples in {bucket_elapsed:.1f}s")

    # Package and upload
    if samples and upload:
        tar_dir = os.path.join(TMP_DIR, "tars")
        url = package_and_upload(samples, dimension, bucket, tar_dir)
        if url:
            mark_bucket_done(dimension, bucket)

    # Cleanup work directory
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return samples


def worker_loop(gpu, work_items=None):
    """Main worker loop: process buckets from a list or all available.

    Args:
        gpu: GPU index (used for port calculation)
        work_items: optional list of (dimension, bucket) tuples to process
    """
    print(f"Worker starting on GPU {gpu}", flush=True)

    if work_items is None:
        from dataset_loader import get_all_available_dimension_buckets
        work_items = list(get_all_available_dimension_buckets())

    total = len(work_items)
    for idx, (dim_name, bucket) in enumerate(work_items):
        print(f"\n[{idx+1}/{total}] Next: {dim_name} {bucket_to_str(bucket)}", flush=True)
        try:
            process_bucket(dim_name, bucket, gpu)
        except Exception as e:
            print(f"Bucket {dim_name}_{bucket_to_str(bucket)} failed: {e}")
            traceback.print_exc()

    print(f"Worker on GPU {gpu} finished", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dimension", type=str, default=None,
                        help="Specific dimension to process")
    parser.add_argument("--bucket", type=str, default=None,
                        help="Specific bucket, e.g. '2to3'")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--echo-port", type=int, default=None)
    parser.add_argument("--vc-port", type=int, default=None)
    parser.add_argument("--ei-port", type=int, default=None)
    parser.add_argument("--queue-file", type=str, default=None,
                        help="JSON file with list of [dimension, [bmin, bmax]] work items")
    args = parser.parse_args()

    if args.echo_port:
        ECHO_PORT_OVERRIDE = args.echo_port
    if args.vc_port:
        VC_PORT_OVERRIDE = args.vc_port
    if args.ei_port:
        EI_PORT_OVERRIDE = args.ei_port

    if args.queue_file:
        with open(args.queue_file) as f:
            items = json.load(f)
        work_items = [(dim, tuple(bucket)) for dim, bucket in items]
        worker_loop(args.gpu, work_items)
    elif args.dimension and args.bucket:
        parts = args.bucket.replace("neg", "-").split("to")
        bucket = (int(parts[0]), int(parts[1]))
        process_bucket(args.dimension, bucket, args.gpu, upload=not args.no_upload)
    elif args.dimension:
        dim_info = ALL_DIMENSIONS[args.dimension]
        for bucket in dim_info["buckets"]:
            process_bucket(args.dimension, bucket, args.gpu, upload=not args.no_upload)
    else:
        worker_loop(args.gpu)
