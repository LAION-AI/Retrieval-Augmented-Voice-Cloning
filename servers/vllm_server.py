#!/usr/bin/env python3
"""
VLLM server launch wrapper for LFM 2.5 1.2B Instruct.

Starts a VLLM OpenAI-compatible API server on a specified GPU.
Usage: python vllm_server.py --gpu 5 --port 9100
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="GPU to use")
    parser.add_argument("--port", type=int, default=9100, help="Port to serve on")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15,
                        help="Fraction of GPU memory for VLLM")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Fix cuDNN
    if "ml-general" in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = ""

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", "2048",
        "--trust-remote-code",
    ]

    print(f"Starting VLLM server: {' '.join(cmd)}", flush=True)
    print(f"  GPU: {args.gpu}, Port: {args.port}", flush=True)

    proc = subprocess.Popen(cmd, env=env)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
