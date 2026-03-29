#!/usr/bin/env python3
"""
Pipeline launcher: starts servers and workers for full data generation.

Optimized for the current GPU allocation:
  - Worker A: Echo TTS (GPU 1, port 9201) + EI (GPU 3, port 9403)
  - Worker B: Echo TTS (GPU 6, port 9206) + EI (GPU 7, port 9407)
  - Shared:   VC (GPU 2, port 9302)

Usage:
  python run_pipeline.py                    # Full run with upload
  python run_pipeline.py --no-upload        # Skip HF upload
  python run_pipeline.py --dimension Anger  # Only Anger buckets
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVERS_DIR = os.path.join(BASE_DIR, "servers")

# Worker configurations: (echo_gpu, echo_port, ei_gpu, ei_port)
WORKER_CONFIGS = [
    {"name": "A", "echo_gpu": 1, "echo_port": 9201, "ei_gpu": 3, "ei_port": 9403},
    {"name": "B", "echo_gpu": 6, "echo_port": 9206, "ei_gpu": 7, "ei_port": 9407},
]
VC_GPU = 2
VC_PORT = 9302

# Track all subprocesses for cleanup
ALL_PROCS = []


def cleanup(*args):
    """Kill all subprocesses on exit."""
    print("\nShutting down all processes...", flush=True)
    for name, proc in ALL_PROCS:
        if proc.poll() is None:
            print(f"  Stopping {name} (PID {proc.pid})...", flush=True)
            proc.terminate()
    for name, proc in ALL_PROCS:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("All processes stopped.", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def start_server(name, cmd, log_file=None):
    """Start a server subprocess."""
    env = os.environ.copy()
    if "ml-general" in env.get("LD_LIBRARY_PATH", ""):
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
    """Wait for server health endpoint."""
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
    """Check if a server is already running on the given port."""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Voice Acting Pipeline Launcher")
    parser.add_argument("--no-upload", action="store_true",
                        help="Don't upload to HuggingFace")
    parser.add_argument("--dimension", type=str, default=None,
                        help="Only process specific dimension")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip LAION voice download")
    args = parser.parse_args()

    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print("VOICE ACTING PIPELINE - OPTIMIZED LAUNCHER", flush=True)
    print("=" * 60, flush=True)
    print(f"Workers: {len(WORKER_CONFIGS)}", flush=True)
    for wc in WORKER_CONFIGS:
        print(f"  Worker {wc['name']}: Echo GPU {wc['echo_gpu']} (:{wc['echo_port']}) + "
              f"EI GPU {wc['ei_gpu']} (:{wc['ei_port']})", flush=True)
    print(f"  Shared VC: GPU {VC_GPU} (:{VC_PORT})", flush=True)
    print(f"Upload: {'No' if args.no_upload else 'Yes'}", flush=True)
    print(f"Dimension: {args.dimension or 'ALL'}", flush=True)
    print()

    # ── Step 1: Download LAION reference voices ────────────────────────────
    if not args.skip_download:
        print("Step 1: Checking LAION reference voices...", flush=True)
        sys.path.insert(0, BASE_DIR)
        from dataset_loader import download_laion_voices, get_laion_voice_paths
        download_laion_voices()
        paths = get_laion_voice_paths()
        print(f"  {len(paths)} reference voices available", flush=True)
    else:
        print("Step 1: Skipping LAION download", flush=True)
    print()

    # ── Step 2: Start servers ──────────────────────────────────────────────
    print("Step 2: Starting servers...", flush=True)

    # VC server (shared)
    if is_server_running(VC_PORT):
        print(f"  VC (:{VC_PORT}): already running", flush=True)
    else:
        start_server(f"vc_gpu{VC_GPU}", [
            sys.executable, os.path.join(SERVERS_DIR, "vc_server.py"),
            "--gpu", str(VC_GPU), "--port", str(VC_PORT),
        ], os.path.join(log_dir, f"vc_gpu{VC_GPU}.log"))

    # Per-worker servers
    for wc in WORKER_CONFIGS:
        # Echo TTS
        if is_server_running(wc["echo_port"]):
            print(f"  Echo TTS (:{wc['echo_port']}): already running", flush=True)
        else:
            start_server(f"echo_gpu{wc['echo_gpu']}", [
                sys.executable, os.path.join(SERVERS_DIR, "echo_tts_server.py"),
                "--gpu", str(wc["echo_gpu"]), "--port", str(wc["echo_port"]),
            ], os.path.join(log_dir, f"echo_gpu{wc['echo_gpu']}.log"))

        # EI
        if is_server_running(wc["ei_port"]):
            print(f"  EI (:{wc['ei_port']}): already running", flush=True)
        else:
            start_server(f"ei_gpu{wc['ei_gpu']}", [
                sys.executable, os.path.join(SERVERS_DIR, "ei_server.py"),
                "--gpu", str(wc["ei_gpu"]), "--port", str(wc["ei_port"]),
            ], os.path.join(log_dir, f"ei_gpu{wc['ei_gpu']}.log"))

    print()

    # ── Step 3: Wait for all servers healthy ───────────────────────────────
    print("Step 3: Waiting for servers...", flush=True)
    all_healthy = True

    if not wait_for_health(f"http://localhost:{VC_PORT}/health", f"VC (:{VC_PORT})", timeout=120):
        all_healthy = False

    for wc in WORKER_CONFIGS:
        if not wait_for_health(f"http://localhost:{wc['echo_port']}/health",
                               f"Echo TTS (:{wc['echo_port']})", timeout=120):
            all_healthy = False
        if not wait_for_health(f"http://localhost:{wc['ei_port']}/health",
                               f"EI (:{wc['ei_port']})", timeout=120):
            all_healthy = False

    if not all_healthy:
        print("\nWARNING: Some servers failed to start. Continuing with available servers.", flush=True)
    print()

    # ── Step 4: Build work queue ───────────────────────────────────────────
    print("Step 4: Building work queue...", flush=True)
    from dataset_loader import get_all_available_dimension_buckets
    from worker import is_bucket_done

    all_items = []
    for dim_name, bucket in get_all_available_dimension_buckets():
        if args.dimension and dim_name != args.dimension:
            continue
        if not is_bucket_done(dim_name, bucket):
            all_items.append((dim_name, list(bucket)))

    print(f"  Total work items: {len(all_items)}", flush=True)
    if not all_items:
        print("  Nothing to do! All buckets already processed.", flush=True)
        return

    # Distribute round-robin across workers
    worker_queues = {wc["name"]: [] for wc in WORKER_CONFIGS}
    worker_names = [wc["name"] for wc in WORKER_CONFIGS]
    for i, item in enumerate(all_items):
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
        queue_file = os.path.join(BASE_DIR, "tmp", f"queue_worker_{wc['name']}.json")
        os.makedirs(os.path.dirname(queue_file), exist_ok=True)
        with open(queue_file, "w") as f:
            json.dump(items, f)

        cmd = [
            sys.executable, os.path.join(BASE_DIR, "worker.py"),
            "--gpu", str(wc["echo_gpu"]),
            "--echo-port", str(wc["echo_port"]),
            "--vc-port", str(VC_PORT),
            "--ei-port", str(wc["ei_port"]),
            "--queue-file", queue_file,
        ]
        if args.no_upload:
            cmd.append("--no-upload")

        log_file = os.path.join(log_dir, f"worker_{wc['name']}.log")
        proc = start_server(f"worker_{wc['name']}", cmd, log_file)
        worker_procs.append((wc["name"], proc, log_file))

    print()
    print("=" * 60, flush=True)
    print("Pipeline running! Monitor logs:", flush=True)
    for name, proc, log_file in worker_procs:
        print(f"  Worker {name}: tail -f {log_file}", flush=True)
    print(f"\nPress Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    # ── Step 6: Monitor workers ────────────────────────────────────────────
    while True:
        all_done = True
        for name, proc, log_file in worker_procs:
            if proc.poll() is None:
                all_done = False
            else:
                ret = proc.returncode
                if ret != 0:
                    print(f"Worker {name} exited with code {ret}", flush=True)

        if all_done:
            break
        time.sleep(10)

    print("\nAll workers finished!", flush=True)

    # Count completed buckets
    from config import PROGRESS_DIR
    done_files = [f for f in os.listdir(PROGRESS_DIR) if f.endswith(".done")] if os.path.exists(PROGRESS_DIR) else []
    print(f"Completed buckets: {len(done_files)}", flush=True)


if __name__ == "__main__":
    main()
