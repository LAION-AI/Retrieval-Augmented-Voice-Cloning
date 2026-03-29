#!/usr/bin/env python3
"""
Scaled pipeline: 1000 samples per bucket for top-2 buckets of all 40 emotions.

4 workers:
  Worker A: Echo GPU 1 (9201) + EI GPU 3 (9403)
  Worker B: Echo GPU 6 (9206) + EI GPU 7 (9407)
  Worker C: Echo GPU 0 (9200) + EI GPU 0 (9400)
  Worker D: Echo GPU 5 (9205) + EI GPU 5 (9405)
  Shared:   VC GPU 2 (9302)

Usage:
  python run_scaled.py
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVERS_DIR = os.path.join(BASE_DIR, "servers")

# 4 worker configs
WORKER_CONFIGS = [
    {"name": "A", "echo_gpu": 1, "echo_port": 9201, "ei_gpu": 3, "ei_port": 9403},
    {"name": "B", "echo_gpu": 6, "echo_port": 9206, "ei_gpu": 7, "ei_port": 9407},
    {"name": "C", "echo_gpu": 0, "echo_port": 9200, "ei_gpu": 0, "ei_port": 9400},
    {"name": "D", "echo_gpu": 5, "echo_port": 9205, "ei_gpu": 5, "ei_port": 9405},
]
VC_PORT = 9302

ALL_PROCS = []

SAMPLES_PER_BUCKET = 1000
UPLOAD_CHUNK_SIZE = 50  # Upload every 50 samples to avoid huge tars

# Use /tmp for working files to avoid filling /home
SCALED_TMP_DIR = "/tmp/voice-pipeline-scaled"


def cleanup(*args):
    print("\nShutting down...", flush=True)
    for name, proc in ALL_PROCS:
        if proc.poll() is None:
            proc.terminate()
    for name, proc in ALL_PROCS:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def start_server(name, cmd, log_file=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    if log_file:
        log_fd = open(log_file, "w")
        proc = subprocess.Popen(cmd, env=env, cwd=BASE_DIR,
                                stdout=log_fd, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(cmd, env=env, cwd=BASE_DIR,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    ALL_PROCS.append((name, proc))
    print(f"  Started {name} (PID {proc.pid})", flush=True)
    return proc


def wait_for_health(url, name, timeout=300, interval=3):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"  {name}: READY", flush=True)
                return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"  {name}: TIMEOUT after {timeout}s", flush=True)
    return False


def is_server_running(port):
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_top2_buckets_for_emotions():
    """For each of the 40 emotions, find the 2 highest available buckets."""
    sys.path.insert(0, BASE_DIR)
    from config import EMOTION_KEYS, discover_available_tars

    tars = discover_available_tars()
    work_items = []

    for emo in EMOTION_KEYS:
        available = tars.get(emo, [])
        if not available:
            print(f"  WARNING: No tars available for {emo}", flush=True)
            continue

        # Parse bucket strings to tuples and sort by max value descending
        buckets = []
        for bs in available:
            parts = bs.split("to")
            try:
                bmin, bmax = int(parts[0]), int(parts[1])
                buckets.append((bmin, bmax))
            except ValueError:
                continue

        # Sort by bucket max descending (highest intensity first)
        buckets.sort(key=lambda b: -b[1])
        top2 = buckets[:2]

        for bucket in top2:
            work_items.append((emo, list(bucket)))

        bstrs = [f"{b[0]}to{b[1]}" for b in top2]
        print(f"  {emo:40s} top-2: {', '.join(bstrs)}", flush=True)

    return work_items


def main():
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(SCALED_TMP_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("SCALED PIPELINE: 1000 samples × top-2 buckets × 40 emotions", flush=True)
    print("=" * 70, flush=True)
    print(f"Workers: {len(WORKER_CONFIGS)}", flush=True)
    for wc in WORKER_CONFIGS:
        print(f"  Worker {wc['name']}: Echo GPU {wc['echo_gpu']} (:{wc['echo_port']}) + "
              f"EI GPU {wc['ei_gpu']} (:{wc['ei_port']})", flush=True)
    print(f"  Shared VC: GPU 2 (:{VC_PORT})", flush=True)
    print(f"  Samples per bucket: {SAMPLES_PER_BUCKET}", flush=True)
    print(f"  Upload chunk size: {UPLOAD_CHUNK_SIZE}", flush=True)
    print(f"  Tmp dir: {SCALED_TMP_DIR}", flush=True)
    print()

    # ── Step 1: Start new servers on GPUs 0 and 5 ────────────────────────
    print("Step 1: Starting servers...", flush=True)

    # Check/start all servers
    for wc in WORKER_CONFIGS:
        # Echo TTS
        if is_server_running(wc["echo_port"]):
            print(f"  Echo TTS (:{wc['echo_port']}): already running", flush=True)
        else:
            # Use compiled version if available
            echo_script = os.path.join(SERVERS_DIR, "echo_tts_server_compiled.py")
            if not os.path.exists(echo_script):
                echo_script = os.path.join(SERVERS_DIR, "echo_tts_server.py")
            cmd = [sys.executable, echo_script,
                   "--gpu", str(wc["echo_gpu"]), "--port", str(wc["echo_port"])]
            if "compiled" in echo_script:
                cmd.append("--compile")
            start_server(f"echo_gpu{wc['echo_gpu']}", cmd,
                        os.path.join(log_dir, f"echo_gpu{wc['echo_gpu']}_scaled.log"))

        # EI
        if is_server_running(wc["ei_port"]):
            print(f"  EI (:{wc['ei_port']}): already running", flush=True)
        else:
            start_server(f"ei_gpu{wc['ei_gpu']}", [
                sys.executable, os.path.join(SERVERS_DIR, "ei_server.py"),
                "--gpu", str(wc["ei_gpu"]), "--port", str(wc["ei_port"]),
            ], os.path.join(log_dir, f"ei_gpu{wc['ei_gpu']}_scaled.log"))

    # VC (shared)
    if is_server_running(VC_PORT):
        print(f"  VC (:{VC_PORT}): already running", flush=True)
    else:
        start_server(f"vc_gpu2", [
            sys.executable, os.path.join(SERVERS_DIR, "vc_server.py"),
            "--gpu", "2", "--port", str(VC_PORT),
        ], os.path.join(log_dir, "vc_gpu2_scaled.log"))

    print()

    # ── Step 2: Wait for all servers ───────────────────────────────────────
    print("Step 2: Waiting for servers...", flush=True)
    all_healthy = True

    if not wait_for_health(f"http://localhost:{VC_PORT}/health", f"VC (:{VC_PORT})", timeout=120):
        all_healthy = False

    for wc in WORKER_CONFIGS:
        if not wait_for_health(f"http://localhost:{wc['echo_port']}/health",
                               f"Echo TTS (:{wc['echo_port']})", timeout=300):
            all_healthy = False
        if not wait_for_health(f"http://localhost:{wc['ei_port']}/health",
                               f"EI (:{wc['ei_port']})", timeout=300):
            all_healthy = False

    if not all_healthy:
        print("\nWARNING: Some servers failed. Check logs.", flush=True)
    print()

    # ── Step 3: Build work queue (top 2 buckets × 40 emotions) ────────────
    print("Step 3: Discovering top-2 buckets for all 40 emotions...", flush=True)
    all_items = get_top2_buckets_for_emotions()
    print(f"\n  Total work items: {len(all_items)} (expecting ~80)", flush=True)

    # Filter out already-done buckets (with 1000 samples)
    # Note: old .done files were for 10 samples. We use a new progress dir.
    scaled_progress = os.path.join(BASE_DIR, "progress_scaled")
    os.makedirs(scaled_progress, exist_ok=True)

    remaining = []
    for dim, bucket in all_items:
        bstr = f"{bucket[0]}to{bucket[1]}"
        done_file = os.path.join(scaled_progress, f"{dim}_{bstr}.done")
        if os.path.exists(done_file):
            print(f"  Skipping {dim}_{bstr} (already done)", flush=True)
        else:
            remaining.append((dim, bucket))

    print(f"  Remaining: {len(remaining)} buckets", flush=True)
    if not remaining:
        print("  Nothing to do!", flush=True)
        return
    print()

    # ── Step 4: Distribute across 4 workers ────────────────────────────────
    print("Step 4: Distributing work...", flush=True)
    worker_queues = {wc["name"]: [] for wc in WORKER_CONFIGS}
    worker_names = [wc["name"] for wc in WORKER_CONFIGS]
    for i, item in enumerate(remaining):
        wname = worker_names[i % len(worker_names)]
        worker_queues[wname].append(item)

    for wname, items in worker_queues.items():
        print(f"  Worker {wname}: {len(items)} buckets", flush=True)
    print()

    # ── Step 5: Launch workers ─────────────────────────────────────────────
    print("Step 5: Launching workers...", flush=True)
    worker_procs = []

    for wc in WORKER_CONFIGS:
        items = worker_queues[wc["name"]]
        if not items:
            continue

        # Write queue file
        queue_file = os.path.join(SCALED_TMP_DIR, f"queue_worker_{wc['name']}.json")
        with open(queue_file, "w") as f:
            json.dump(items, f)

        cmd = [
            sys.executable, os.path.join(BASE_DIR, "worker_scaled.py"),
            "--gpu", str(wc["echo_gpu"]),
            "--echo-port", str(wc["echo_port"]),
            "--vc-port", str(VC_PORT),
            "--ei-port", str(wc["ei_port"]),
            "--queue-file", queue_file,
            "--samples", str(SAMPLES_PER_BUCKET),
            "--chunk-size", str(UPLOAD_CHUNK_SIZE),
            "--tmp-dir", SCALED_TMP_DIR,
            "--progress-dir", scaled_progress,
        ]

        log_file = os.path.join(log_dir, f"worker_{wc['name']}_scaled.log")
        proc = start_server(f"worker_{wc['name']}", cmd, log_file)
        worker_procs.append((wc["name"], proc, log_file))

    print()
    print("=" * 70, flush=True)
    print("SCALED PIPELINE RUNNING!", flush=True)
    print(f"  {len(remaining)} buckets × {SAMPLES_PER_BUCKET} samples = "
          f"{len(remaining) * SAMPLES_PER_BUCKET} total samples", flush=True)
    print(f"  = {len(remaining) * SAMPLES_PER_BUCKET * 6} WAV generations", flush=True)
    print()
    print("Monitor logs:", flush=True)
    for name, proc, log_file in worker_procs:
        print(f"  Worker {name}: tail -f {log_file}", flush=True)
    print(f"\nPress Ctrl+C to stop.", flush=True)
    print("=" * 70, flush=True)

    # ── Step 6: Monitor ────────────────────────────────────────────────────
    while True:
        all_done = True
        for name, proc, log_file in worker_procs:
            if proc.poll() is None:
                all_done = False
        if all_done:
            break
        time.sleep(30)

    print("\nAll workers finished!", flush=True)
    done_files = [f for f in os.listdir(scaled_progress) if f.endswith(".done")]
    print(f"Completed buckets: {len(done_files)}", flush=True)


if __name__ == "__main__":
    main()
