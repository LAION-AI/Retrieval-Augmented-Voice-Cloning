"""
WebDataset tar packaging and HuggingFace upload.

Packages samples into tar files and uploads them to a HuggingFace dataset repo.
"""

import io
import json
import os
import random
import tarfile
import time

from huggingface_hub import HfApi

from config import HF_UPLOAD_REPO, bucket_to_str


def create_sample_tar_entry(tar, key, ext, data):
    """Add a file to a tar archive.

    Args:
        tar: tarfile.TarFile object
        key: base key (e.g. 'sample_001')
        ext: extension (e.g. '.json', '.emotional_seed0.wav')
        data: bytes or str data
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    info = tarfile.TarInfo(name=f"{key}{ext}")
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def package_bucket_samples(samples, dimension, bucket, output_dir):
    """Package a list of samples into a WebDataset tar file.

    Args:
        samples: list of sample dicts, each containing:
            - sample_id: unique identifier
            - emotional_wavs: list of (seed, wav_path) tuples (3 per sample)
            - neutral_wavs: list of (seed, wav_path) tuples (3 per sample)
            - ref_audio_path: path to reference speaker audio
            - metadata: dict with all metadata (topic, letter, scores, etc.)
        dimension: dimension name
        bucket: bucket tuple
        output_dir: directory to save tar

    Returns:
        path to created tar file
    """
    os.makedirs(output_dir, exist_ok=True)

    bucket_str = bucket_to_str(bucket)
    random_id = random.randint(1000000000, 9999999999)
    tar_name = f"{dimension}_{bucket_str}_{random_id}.tar"
    tar_path = os.path.join(output_dir, tar_name)

    with tarfile.open(tar_path, "w") as tar:
        for sample in samples:
            sid = sample["sample_id"]

            # Emotional WAVs (3 seeds)
            for seed, wav_path in sample["emotional_wavs"]:
                with open(wav_path, "rb") as f:
                    create_sample_tar_entry(tar, sid, f".emotional_seed{seed}.wav", f.read())

            # Neutral WAVs (3 seeds)
            for seed, wav_path in sample["neutral_wavs"]:
                with open(wav_path, "rb") as f:
                    create_sample_tar_entry(tar, sid, f".neutral_seed{seed}.wav", f.read())

            # Reference audio
            if sample.get("ref_audio_path") and os.path.exists(sample["ref_audio_path"]):
                with open(sample["ref_audio_path"], "rb") as f:
                    create_sample_tar_entry(tar, sid, ".ref_audio.wav", f.read())

            # Metadata JSON
            metadata_json = json.dumps(sample["metadata"], indent=2, ensure_ascii=False)
            create_sample_tar_entry(tar, sid, ".json", metadata_json)

    print(f"Created tar: {tar_path} ({os.path.getsize(tar_path) / 1024:.1f} KB)")
    return tar_path


def upload_tar_to_hf(tar_path, repo_id=HF_UPLOAD_REPO, path_in_repo=None):
    """Upload a tar file to HuggingFace dataset repo.

    Args:
        tar_path: local path to tar file
        repo_id: HuggingFace repo ID
        path_in_repo: path within repo (default: data/{tar_filename})

    Returns:
        URL of uploaded file
    """
    api = HfApi()
    tar_name = os.path.basename(tar_path)

    if path_in_repo is None:
        path_in_repo = f"data/{tar_name}"

    print(f"Uploading {tar_name} to {repo_id}...", flush=True)
    t0 = time.time()

    try:
        url = api.upload_file(
            path_or_fileobj=tar_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        elapsed = round(time.time() - t0, 1)
        print(f"Uploaded {tar_name} in {elapsed}s: {url}", flush=True)
        return url
    except Exception as e:
        print(f"Upload failed for {tar_name}: {e}", flush=True)
        raise


def package_and_upload(samples, dimension, bucket, output_dir,
                       repo_id=HF_UPLOAD_REPO, delete_after=True):
    """Package samples into tar and upload to HuggingFace.

    Returns upload URL or None on failure.
    """
    try:
        tar_path = package_bucket_samples(samples, dimension, bucket, output_dir)
        url = upload_tar_to_hf(tar_path, repo_id=repo_id)

        if delete_after and url:
            os.remove(tar_path)
            print(f"Deleted local tar: {tar_path}")

        return url
    except Exception as e:
        print(f"Package/upload failed: {e}")
        return None
