#!/usr/bin/env python3
"""
FastAPI server for ChatterboxVC voice conversion.

Manages a persistent subprocess running under spiritvenv Python 3.13.
Usage: python vc_server.py --gpu 6 --port 9306
"""

import os
import sys
import argparse
import asyncio
import json
import time
import tempfile

# Fix cuDNN
if "ml-general" in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = ""
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="ChatterboxVC Server")

SPIRITVENV_PYTHON = os.environ.get("SPIRITVENV_PYTHON", os.path.expanduser("~/spiritvenv/bin/python"))
CHATTERBOX_SR = 24000


class State:
    vc_process = None
    vc_lock = None
    gpu = None


state = State()


VC_WORKER_CODE = '''
import os
import sys
import json
import time
import traceback

# Capture stdout for protocol before any imports
_proto_fd = os.dup(sys.stdout.fileno())
_proto_out = os.fdopen(_proto_fd, "w", buffering=1)
sys.stdout = sys.stderr

def proto_send(obj):
    _proto_out.write(json.dumps(obj) + "\\n")
    _proto_out.flush()

DEVICE = "cuda:{gpu}"

def load_vc():
    from chatterbox import ChatterboxVC
    proto_send({{"status": "loading", "msg": "Loading ChatterboxVC..."}})
    model = ChatterboxVC.from_pretrained(device=DEVICE)
    proto_send({{"status": "ready", "msg": "ChatterboxVC loaded on " + DEVICE}})
    return model

def main():
    vc = load_vc()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            source = req["source"]
            target = req["target"]
            output = req["output"]
            t0 = time.time()
            wav = vc.generate(audio=source, target_voice_path=target)
            sr = vc.sr
            import torchaudio
            torchaudio.save(output, wav.cpu().float(), sr)
            elapsed = time.time() - t0
            proto_send({{
                "status": "ok",
                "output": output,
                "sample_rate": sr,
                "elapsed": round(elapsed, 3),
            }})
        except Exception as e:
            proto_send({{
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }})

if __name__ == "__main__":
    main()
'''


VC_TIMEOUT = 60  # Kill subprocess if a single conversion takes longer than this


async def _read_vc_json_line(timeout=VC_TIMEOUT):
    """Read JSON line from VC subprocess, skipping non-JSON output."""
    while True:
        line = await asyncio.wait_for(state.vc_process.stdout.readline(), timeout=timeout)
        text = line.decode().strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"  VC (non-JSON): {text}", flush=True)
            continue


async def _kill_subprocess():
    """Kill the VC subprocess forcefully."""
    if state.vc_process is not None:
        pid = state.vc_process.pid
        print(f"  Killing stuck VC subprocess (PID {pid})...", flush=True)
        try:
            state.vc_process.kill()
            await asyncio.wait_for(state.vc_process.wait(), timeout=5)
        except Exception:
            pass
        state.vc_process = None
        print(f"  VC subprocess killed.", flush=True)


async def ensure_vc_subprocess():
    """Start VC subprocess if not running."""
    if state.vc_process is not None and state.vc_process.returncode is None:
        return

    state.vc_process = None
    print(f"Starting VC subprocess on GPU {state.gpu}...", flush=True)

    # Write worker script to temp file
    worker_code = VC_WORKER_CODE.format(gpu=state.gpu)
    worker_path = os.path.join(tempfile.gettempdir(), f"vc_worker_gpu{state.gpu}.py")
    with open(worker_path, "w") as f:
        f.write(worker_code)

    state.vc_process = await asyncio.create_subprocess_exec(
        SPIRITVENV_PYTHON, worker_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait for ready (longer timeout for model loading)
    msg = await _read_vc_json_line(timeout=300)
    print(f"  VC subprocess: {msg}", flush=True)
    while msg.get("status") != "ready":
        msg = await _read_vc_json_line(timeout=300)
        print(f"  VC subprocess: {msg}", flush=True)
    state.total_conversions = 0
    state.total_restarts = getattr(state, "total_restarts", 0)


async def vc_convert(source_path, target_path, output_path):
    """Send voice conversion request to subprocess. Auto-restarts on hang."""
    async with state.vc_lock:
        await ensure_vc_subprocess()
        req = json.dumps({
            "source": source_path,
            "target": target_path,
            "output": output_path,
        }) + "\n"
        state.vc_process.stdin.write(req.encode())
        await state.vc_process.stdin.drain()
        try:
            result = await _read_vc_json_line(timeout=VC_TIMEOUT)
            state.total_conversions = getattr(state, "total_conversions", 0) + 1
            return result
        except asyncio.TimeoutError:
            print(f"  VC subprocess hung (>{VC_TIMEOUT}s), killing and restarting...",
                  flush=True)
            await _kill_subprocess()
            state.total_restarts = getattr(state, "total_restarts", 0) + 1
            # Restart subprocess for next request
            await ensure_vc_subprocess()
            raise  # Let caller handle the timeout


@app.on_event("startup")
async def startup():
    state.vc_lock = asyncio.Lock()


@app.get("/health")
async def health():
    running = state.vc_process is not None and state.vc_process.returncode is None
    return {
        "status": "ok",
        "subprocess_running": running,
        "gpu": state.gpu,
        "total_conversions": getattr(state, "total_conversions", 0),
        "total_restarts": getattr(state, "total_restarts", 0),
    }


@app.post("/convert")
async def convert(
    source_path: str = Form(...),
    target_path: str = Form(...),
):
    """Convert source audio to target speaker's voice.

    Args:
        source_path: Path to source audio WAV
        target_path: Path to target speaker audio WAV (identity to clone)

    Returns:
        JSON with output_path, sample_rate, elapsed
    """
    try:
        output_dir = tempfile.mkdtemp(prefix="vc_")
        output_path = os.path.join(output_dir, "converted.wav")

        result = await vc_convert(source_path, target_path, output_path)

        if result.get("status") != "ok":
            return JSONResponse(
                {"error": result.get("error", "unknown"), "traceback": result.get("traceback", "")},
                status_code=500,
            )

        return {
            "status": "ok",
            "output_path": output_path,
            "sample_rate": result.get("sample_rate", CHATTERBOX_SR),
            "elapsed": result.get("elapsed", 0),
        }

    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"VC subprocess hung (>{VC_TIMEOUT}s), killed and restarted",
             "status": "timeout_restart"},
            status_code=503,
        )

    except Exception as e:
        import traceback as tb
        return JSONResponse(
            {"error": str(e), "traceback": tb.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    state.gpu = args.gpu
    print(f"VC server starting on GPU {args.gpu}, port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
