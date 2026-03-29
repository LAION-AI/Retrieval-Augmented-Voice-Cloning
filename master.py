#!/usr/bin/env python3
"""
Master orchestrator for the Voice Acting Pipeline.

Spawns all servers (VLLM, Echo TTS, VC, EI) and workers.
Usage: python master.py --gpus 5,6,7 --hf-repo TTS-AGI/voice-acting-pipeline-output
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import requests

from config import (
    GPUS, VLLM_PORT, LFM_MODEL, HF_UPLOAD_REPO, BASE_DIR,
    echo_tts_port, vc_port, ei_port, ALL_DIMENSIONS,
)


class ProcessManager:
    """Manage server and worker subprocesses."""

    def __init__(self):
        self.processes = {}  # name -> subprocess.Popen
        self.shutting_down = False

    def start(self, name, cmd, env=None, cwd=None):
        """Start a subprocess."""
        if env is None:
            env = os.environ.copy()
        # Fix cuDNN
        if "ml-general" in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = ""

        print(f"  Starting {name}: {' '.join(cmd[:4])}...", flush=True)
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd or BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.processes[name] = proc
        return proc

    def stop_all(self):
        """Stop all processes."""
        self.shutting_down = True
        print("\nStopping all processes...", flush=True)
        for name, proc in self.processes.items():
            if proc.poll() is None:
                print(f"  Terminating {name}...", flush=True)
                proc.terminate()
        # Wait for graceful shutdown
        for name, proc in self.processes.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("All processes stopped.", flush=True)


def wait_for_server(url, name, timeout=300, interval=5):
    """Wait for a server to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"  {name} is ready!", flush=True)
                return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"  WARNING: {name} did not become ready within {timeout}s", flush=True)
    return False


def wait_for_vllm(port, timeout=300):
    """Wait for VLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if resp.status_code == 200:
                print(f"  VLLM is ready!", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"  WARNING: VLLM did not become ready within {timeout}s", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description="Voice Acting Pipeline Master")
    parser.add_argument("--gpus", type=str, default=",".join(map(str, GPUS)),
                        help="Comma-separated GPU IDs (default: from config)")
    parser.add_argument("--hf-repo", type=str, default=HF_UPLOAD_REPO,
                        help="HuggingFace repo for upload")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Skip starting VLLM (if already running)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Don't upload to HuggingFace")
    parser.add_argument("--dimension", type=str, default=None,
                        help="Only process specific dimension")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.15,
                        help="VLLM GPU memory utilization")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"Voice Acting Pipeline Master", flush=True)
    print(f"  GPUs: {gpus}", flush=True)
    print(f"  HF Repo: {args.hf_repo}", flush=True)
    print(f"  Dimensions: {args.dimension or 'ALL'}", flush=True)
    print()

    pm = ProcessManager()

    # Handle signals
    def signal_handler(sig, frame):
        pm.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    servers_dir = os.path.join(BASE_DIR, "servers")

    try:
        # 1. Start VLLM server (shared, on first GPU)
        if not args.no_vllm:
            print("Starting VLLM server...", flush=True)
            pm.start("vllm", [
                sys.executable, os.path.join(servers_dir, "vllm_server.py"),
                "--gpu", str(gpus[0]),
                "--port", str(VLLM_PORT),
                "--model", LFM_MODEL,
                "--gpu-memory-utilization", str(args.vllm_gpu_util),
            ])

        # 2. Start per-GPU servers
        for gpu in gpus:
            print(f"\nStarting servers on GPU {gpu}...", flush=True)

            # Echo TTS
            pm.start(f"echo_tts_gpu{gpu}", [
                sys.executable, os.path.join(servers_dir, "echo_tts_server.py"),
                "--gpu", str(gpu),
                "--port", str(echo_tts_port(gpu)),
            ])

            # VC
            pm.start(f"vc_gpu{gpu}", [
                sys.executable, os.path.join(servers_dir, "vc_server.py"),
                "--gpu", str(gpu),
                "--port", str(vc_port(gpu)),
            ])

            # EI
            pm.start(f"ei_gpu{gpu}", [
                sys.executable, os.path.join(servers_dir, "ei_server.py"),
                "--gpu", str(gpu),
                "--port", str(ei_port(gpu)),
            ])

        # 3. Wait for all servers
        print("\nWaiting for servers to be ready...", flush=True)

        if not args.no_vllm:
            wait_for_vllm(VLLM_PORT, timeout=300)

        for gpu in gpus:
            wait_for_server(
                f"http://localhost:{echo_tts_port(gpu)}/health",
                f"Echo TTS GPU {gpu}", timeout=60)
            wait_for_server(
                f"http://localhost:{vc_port(gpu)}/health",
                f"VC GPU {gpu}", timeout=60)
            wait_for_server(
                f"http://localhost:{ei_port(gpu)}/health",
                f"EI GPU {gpu}", timeout=60)

        # 4. Build work queue (from actually available tars)
        from dataset_loader import get_all_available_dimension_buckets
        work_items = []
        for dim_name, bucket in get_all_available_dimension_buckets():
            if args.dimension and dim_name != args.dimension:
                continue
            work_items.append((dim_name, bucket))

        print(f"\nTotal work items: {len(work_items)}", flush=True)

        # 5. Distribute work across GPUs (round-robin)
        gpu_queues = {gpu: [] for gpu in gpus}
        for i, item in enumerate(work_items):
            gpu = gpus[i % len(gpus)]
            gpu_queues[gpu].append(item)

        # 6. Start workers
        print("\nStarting workers...", flush=True)
        worker_procs = []
        for gpu, items in gpu_queues.items():
            if not items:
                continue
            print(f"  GPU {gpu}: {len(items)} buckets", flush=True)

            # Write queue to temp file
            queue_file = os.path.join(BASE_DIR, "tmp", f"queue_gpu{gpu}.json")
            os.makedirs(os.path.dirname(queue_file), exist_ok=True)
            with open(queue_file, "w") as f:
                json.dump(items, f)

            worker_cmd = [
                sys.executable, os.path.join(BASE_DIR, "worker_runner.py"),
                "--gpu", str(gpu),
                "--queue-file", queue_file,
            ]
            if args.no_upload:
                worker_cmd.append("--no-upload")

            proc = pm.start(f"worker_gpu{gpu}", worker_cmd)
            worker_procs.append(proc)

        # 7. Wait for workers to finish
        print("\nWorkers running. Press Ctrl+C to stop.", flush=True)
        for proc in worker_procs:
            proc.wait()

        print("\nAll workers finished!", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted!", flush=True)
    finally:
        pm.stop_all()


if __name__ == "__main__":
    import json
    main()
