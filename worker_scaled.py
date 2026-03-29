"""
Scaled worker: processes large batches (1000 samples/bucket) with chunked uploads.

Key differences from worker.py:
- Configurable samples-per-bucket (default 1000) and chunk-size (default 50)
- Uploads every chunk_size samples as a separate tar to avoid disk exhaustion
- Uses configurable tmp-dir and progress-dir
- Reads work queue from a JSON file
"""

import json
import os
import random
import shutil
import string
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

from config import (
    SEEDS_PER_SAMPLE, ECHO_TTS_STEPS,
    WORD_COUNT_MIN, WORD_COUNT_MAX, SPEAKER_REF_MAX_DURATION,
    ECHO_TTS_SR, HF_UPLOAD_REPO,
    bucket_to_str, NO_VC_DIMENSIONS,
)
from dataset_loader import (
    get_emotion_samples, decode_sample_to_wav,
    get_random_laion_voice, load_wav, save_wav, resample_audio,
)
from sentence_generator import (
    generate_sentence, sample_punctuation_params, get_random_topic,
)
from uploader import package_bucket_samples, upload_tar_to_hf


# Port overrides (set from CLI args)
ECHO_PORT = None
VC_PORT = None
EI_PORT = None

# Thread pools for pipelining
_ei_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ei")
_sentence_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")


def call_echo_tts(text, ref_audio_path, seed, num_steps=ECHO_TTS_STEPS):
    resp = requests.post(
        f"http://localhost:{ECHO_PORT}/generate",
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


def call_vc(source_path, target_path):
    resp = requests.post(
        f"http://localhost:{VC_PORT}/convert",
        data={
            "source_path": source_path,
            "target_path": target_path,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def call_ei(audio_path):
    resp = requests.post(
        f"http://localhost:{EI_PORT}/score",
        data={"audio_path": audio_path},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def process_sample(sample_idx, emotion_refs, dimension, bucket, work_dir):
    """Process a single sample with pipelined TTS+EI."""
    sample_dir = os.path.join(work_dir, f"sample_{sample_idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Pick random emotion reference
    ref_sample = random.choice(emotion_refs)
    ref_meta = ref_sample["json"]

    # Decode emotion reference to WAV (DACVAE native 48kHz)
    ref_wav_path, ref_sr = decode_sample_to_wav(ref_sample, sample_dir)

    # 2. Roll d10: 10% keep original, 90% VC to LAION ref speaker
    skip_vc = dimension in NO_VC_DIMENSIONS
    use_original_voice = skip_vc or random.random() < 0.1
    speaker_ref_path = ref_wav_path
    vc_info = {"used_vc": False, "skipped_reason": "pitch/age dimension" if skip_vc else None}

    if not use_original_voice:
        try:
            laion_voice_path = get_random_laion_voice()
            vc_result = call_vc(ref_wav_path, laion_voice_path)
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

    # Trim speaker ref to max duration
    spk_duration = spk_audio.shape[-1] / ECHO_TTS_SR
    if spk_duration > SPEAKER_REF_MAX_DURATION:
        max_samples = int(SPEAKER_REF_MAX_DURATION * ECHO_TTS_SR)
        save_wav(spk_ref_441_path, spk_audio[..., :max_samples], ECHO_TTS_SR)

    # 3. Generate emotional + neutral sentences concurrently
    emo_topic = get_random_topic()
    emo_letter = random.choice(string.ascii_uppercase)
    emo_word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)
    emo_punct = sample_punctuation_params()

    neu_topic = emo_topic
    neu_letter = random.choice([l for l in string.ascii_uppercase if l != emo_letter])
    neu_word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)

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

    # 4. Pipelined TTS + EI
    emotional_wavs = []
    neutral_wavs = []
    emotional_results = []
    neutral_results = []
    ei_futures = []

    seeds = [random.randint(0, 999999) for _ in range(SEEDS_PER_SAMPLE)]

    for seed in seeds:
        # Emotional TTS
        try:
            emo_result = call_echo_tts(emo_sentence["text"], spk_ref_441_path, seed)
            if emo_result.get("status") == "ok":
                emotional_wavs.append((seed, emo_result["output_path"]))
                gen = {
                    "seed": seed,
                    "path": emo_result["output_path"],
                    "duration": emo_result.get("duration", 0),
                    "elapsed": emo_result.get("elapsed", 0),
                }
                emotional_results.append(gen)
                fut = _ei_pool.submit(call_ei, gen["path"])
                ei_futures.append((fut, gen, "emotional", emo_sentence["text"]))
        except Exception as e:
            print(f"  Echo TTS emotional seed={seed} failed: {e}")

        # Neutral TTS
        try:
            neu_result = call_echo_tts(neu_sentence["text"], spk_ref_441_path, seed)
            if neu_result.get("status") == "ok":
                neutral_wavs.append((seed, neu_result["output_path"]))
                gen = {
                    "seed": seed,
                    "path": neu_result["output_path"],
                    "duration": neu_result.get("duration", 0),
                    "elapsed": neu_result.get("elapsed", 0),
                }
                neutral_results.append(gen)
                fut = _ei_pool.submit(call_ei, gen["path"])
                ei_futures.append((fut, gen, "neutral", neu_sentence["text"]))
        except Exception as e:
            print(f"  Echo TTS neutral seed={seed} failed: {e}")

    # 5. Collect EI results
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
    sample_id = f"{dimension}_{bucket_str}_{sample_idx:04d}"

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


def upload_chunk(chunk_samples, dimension, bucket, tar_dir):
    """Package a chunk of samples into a tar and upload to HF."""
    try:
        tar_path = package_bucket_samples(chunk_samples, dimension, bucket, tar_dir)
        url = upload_tar_to_hf(tar_path, repo_id=HF_UPLOAD_REPO)
        if url:
            os.remove(tar_path)
            print(f"  Uploaded + deleted chunk tar", flush=True)
        return url
    except Exception as e:
        print(f"  Chunk upload failed: {e}", flush=True)
        traceback.print_exc()
        return None


def process_bucket_scaled(dimension, bucket, num_samples, chunk_size,
                          tmp_dir, progress_dir):
    """Process a bucket with chunked uploads.

    Generates num_samples total, uploading every chunk_size samples.
    """
    bucket_str = bucket_to_str(bucket)
    tag = f"{dimension}_{bucket_str}"

    # Check if already done
    done_file = os.path.join(progress_dir, f"{tag}.done")
    if os.path.exists(done_file):
        print(f"  Skipping {tag} (already done)", flush=True)
        return

    # Check partial progress
    partial_file = os.path.join(progress_dir, f"{tag}.partial")
    start_idx = 0
    if os.path.exists(partial_file):
        try:
            with open(partial_file) as f:
                start_idx = int(f.read().strip())
            print(f"  Resuming {tag} from sample {start_idx}", flush=True)
        except Exception:
            start_idx = 0

    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {tag} ({num_samples} samples, chunk={chunk_size})", flush=True)
    print(f"{'='*60}", flush=True)

    # Download emotion references
    t0 = time.time()
    emotion_refs = get_emotion_samples(dimension, bucket)
    if not emotion_refs:
        print(f"  No samples found for {tag}, skipping", flush=True)
        return
    print(f"  Loaded {len(emotion_refs)} emotion refs in {time.time()-t0:.1f}s", flush=True)

    work_dir = os.path.join(tmp_dir, tag)
    tar_dir = os.path.join(tmp_dir, "tars")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(tar_dir, exist_ok=True)

    chunk_samples = []
    total_uploaded = 0
    bucket_start = time.time()

    for i in range(start_idx, num_samples):
        sample_num = i + 1
        print(f"\n  [{tag}] Sample {sample_num}/{num_samples}", flush=True)
        t0 = time.time()

        try:
            sample = process_sample(i, emotion_refs, dimension, bucket, work_dir)
            chunk_samples.append(sample)
            elapsed = time.time() - t0
            n_emo = len(sample["emotional_wavs"])
            n_neu = len(sample["neutral_wavs"])

            done_so_far = i - start_idx + 1
            rate = done_so_far / (time.time() - bucket_start)
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"    Done in {elapsed:.1f}s ({n_emo} emo + {n_neu} neu) "
                  f"| {done_so_far}/{num_samples - start_idx} | "
                  f"ETA: {eta/60:.1f}min", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            traceback.print_exc()

        # Upload chunk when full
        if len(chunk_samples) >= chunk_size:
            print(f"\n  Uploading chunk ({len(chunk_samples)} samples)...", flush=True)
            url = upload_chunk(chunk_samples, dimension, bucket, tar_dir)
            if url:
                total_uploaded += len(chunk_samples)
                # Save partial progress
                with open(partial_file, "w") as f:
                    f.write(str(i + 1))

            # Cleanup sample work dirs for this chunk
            for s in chunk_samples:
                try:
                    if s.get("work_dir") and os.path.exists(s["work_dir"]):
                        shutil.rmtree(s["work_dir"])
                except Exception:
                    pass
            chunk_samples = []

    # Upload remaining samples
    if chunk_samples:
        print(f"\n  Uploading final chunk ({len(chunk_samples)} samples)...", flush=True)
        url = upload_chunk(chunk_samples, dimension, bucket, tar_dir)
        if url:
            total_uploaded += len(chunk_samples)
        for s in chunk_samples:
            try:
                if s.get("work_dir") and os.path.exists(s["work_dir"]):
                    shutil.rmtree(s["work_dir"])
            except Exception:
                pass

    # Cleanup work dir
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    # Mark done (only if we actually generated samples)
    total_time = time.time() - bucket_start
    if total_uploaded > 0:
        print(f"\n  {tag}: COMPLETE - {total_uploaded} samples uploaded in "
              f"{total_time/60:.1f}min", flush=True)
        with open(done_file, "w") as f:
            f.write(json.dumps({
                "samples": total_uploaded,
                "time": round(total_time, 1),
                "timestamp": time.time(),
            }))
        # Remove partial progress file
        if os.path.exists(partial_file):
            os.remove(partial_file)
    else:
        print(f"\n  {tag}: FAILED - 0 samples generated in "
              f"{total_time/60:.1f}min, NOT marking as done", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scaled worker with chunked uploads")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--echo-port", type=int, required=True)
    parser.add_argument("--vc-port", type=int, required=True)
    parser.add_argument("--ei-port", type=int, required=True)
    parser.add_argument("--queue-file", type=str, required=True,
                        help="JSON file: list of [dimension, [bmin, bmax]]")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Samples per bucket (default 1000)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Upload every N samples (default 50)")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/voice-pipeline-scaled")
    parser.add_argument("--progress-dir", type=str, required=True)
    args = parser.parse_args()

    global ECHO_PORT, VC_PORT, EI_PORT
    ECHO_PORT = args.echo_port
    VC_PORT = args.vc_port
    EI_PORT = args.ei_port

    with open(args.queue_file) as f:
        items = json.load(f)
    work_items = [(dim, tuple(bucket)) for dim, bucket in items]

    print(f"Worker GPU {args.gpu}: {len(work_items)} buckets, "
          f"{args.samples} samples/bucket, chunk={args.chunk_size}", flush=True)
    print(f"  Echo: :{ECHO_PORT}, VC: :{VC_PORT}, EI: :{EI_PORT}", flush=True)
    print(f"  Tmp: {args.tmp_dir}, Progress: {args.progress_dir}", flush=True)

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.progress_dir, exist_ok=True)

    for idx, (dim, bucket) in enumerate(work_items):
        print(f"\n[{idx+1}/{len(work_items)}] Next: {dim} {bucket_to_str(bucket)}",
              flush=True)
        try:
            process_bucket_scaled(
                dim, bucket,
                num_samples=args.samples,
                chunk_size=args.chunk_size,
                tmp_dir=args.tmp_dir,
                progress_dir=args.progress_dir,
            )
        except Exception as e:
            print(f"Bucket {dim}_{bucket_to_str(bucket)} failed: {e}", flush=True)
            traceback.print_exc()

    print(f"\nWorker GPU {args.gpu}: ALL DONE", flush=True)


if __name__ == "__main__":
    main()
