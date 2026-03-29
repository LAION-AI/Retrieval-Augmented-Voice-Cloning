#!/usr/bin/env python3
"""
FastAPI server for Empathic Insight Voice+ scoring and captioning.

Loads Whisper encoder + 59 MLP experts on a specified GPU.
Usage: python ei_server.py --gpu 7 --port 9407
"""

import os
import sys
import argparse
import time
from collections import OrderedDict
from pathlib import Path

# Fix cuDNN
if "ml-general" in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = ""
# Disable torch dynamo to prevent import issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch.backends.cudnn.enabled = False

import torch.nn as nn
import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="Empathic Insight Server")

# ─── Config ──────────────────────────────────────────────────────────────────
WHISPER_MODEL_ID = "laion/BUD-E-Whisper"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Plus"
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]
POOLED_DIM = 3072
QUALITY_EXPERT_FILES = {
    "model_score_overall_quality_best.pth",
    "model_score_speech_quality_best.pth",
    "model_score_background_quality_best.pth",
    "model_score_content_enjoyment_best.pth",
}
MODELS_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_cache")


# ─── MLP Architectures ──────────────────────────────────────────────────────

class FullEmbeddingMLP(nn.Module):
    def __init__(self, seq_len, embed_dim, projection_dim, mlp_hidden_dims, mlp_dropout_rates):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i + 1])])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))


class PooledEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, projection_dim, mlp_hidden_dims, mlp_dropout_rates):
        super().__init__()
        self.proj = nn.Linear(input_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i + 1])])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.proj(x))


def pool_embedding(embedding):
    """Pool encoder output [B, seq_len, 768] -> [B, 3072]."""
    return torch.cat([
        embedding.mean(dim=1), embedding.min(dim=1).values,
        embedding.max(dim=1).values, embedding.std(dim=1),
    ], dim=1)


def sanitize_caption(text):
    """Fix BUD-E-Whisper decoder artifact with spurious leading capitals."""
    if len(text) < 3:
        return text
    prefix = text[:3]
    upper_positions = [i for i, c in enumerate(prefix) if c.isupper()]
    if len(upper_positions) >= 2:
        return text[upper_positions[1]:]
    return text


# ─── State ───────────────────────────────────────────────────────────────────

class State:
    whisper = None
    processor = None
    emotion_models = None
    quality_models = None
    device = None
    loading = False


state = State()


def load_models(device):
    """Load Whisper encoder + all MLP experts."""
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from huggingface_hub import snapshot_download

    print(f"Loading Empathic Insight on {device}...", flush=True)

    # Load Whisper
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch.float16,
        cache_dir=MODELS_CACHE, attn_implementation="sdpa")
    whisper_model.to(device).eval()
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID, cache_dir=MODELS_CACHE)

    # Download MLP experts
    mlp_dir = Path(os.path.join(MODELS_CACHE, "empathic_insight_models"))
    mlp_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id=HF_MLP_REPO_ID, local_dir=str(mlp_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.mp3", "*.md", ".gitattributes"])

    emotion_models = {}
    quality_models = {}
    for model_path in mlp_dir.glob("*.pth"):
        filename = model_path.name
        stem = model_path.stem
        parts = stem.split("_")
        dimension_name = "_".join(parts[1:-1]) if "best" in parts[-1] else "_".join(parts[1:])
        is_quality = filename in QUALITY_EXPERT_FILES

        if is_quality:
            mlp_model = PooledEmbeddingMLP(POOLED_DIM, PROJECTION_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUTS).to(device)
        else:
            mlp_model = FullEmbeddingMLP(WHISPER_SEQ_LEN, WHISPER_EMBED_DIM, PROJECTION_DIM, MLP_HIDDEN_DIMS, MLP_DROPOUTS).to(device)

        sd = torch.load(model_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd.items())
        mlp_model.load_state_dict(sd)
        mlp_model.eval().half()

        if is_quality:
            quality_models[dimension_name] = mlp_model
        else:
            emotion_models[dimension_name] = mlp_model

    print(f"Loaded {len(emotion_models)} emotion + {len(quality_models)} quality experts", flush=True)
    return whisper_model, processor, emotion_models, quality_models


@app.on_event("startup")
async def startup():
    pass  # Lazy load


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.whisper is not None, "device": str(state.device)}


@app.post("/score")
async def score(audio_path: str = Form(...)):
    """Score an audio file with all Empathic Insight experts.

    Returns dict of dimension_name -> score, plus a caption.
    """
    try:
        # Lazy load
        if state.whisper is None:
            if state.loading:
                for _ in range(240):
                    time.sleep(0.5)
                    if state.whisper is not None:
                        break
                if state.whisper is None:
                    return JSONResponse({"error": "Models still loading"}, status_code=503)
            else:
                state.loading = True
                try:
                    state.whisper, state.processor, state.emotion_models, state.quality_models = \
                        load_models(state.device)
                finally:
                    state.loading = False

        t0 = time.time()

        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        max_len = 30 * 16000
        if len(audio) > max_len:
            audio = audio[:max_len]

        inputs = state.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(state.device).half()

        with torch.no_grad():
            # Get encoder embeddings
            encoder_outputs = state.whisper.get_encoder()(input_features, return_dict=True)
            embeddings = encoder_outputs.last_hidden_state  # [1, 1500, 768]
            pooled = pool_embedding(embeddings)

            # Score all dimensions
            scores = {}
            for name, model in state.emotion_models.items():
                scores[name] = round(float(model(embeddings).item()), 4)
            for name, model in state.quality_models.items():
                scores[name] = round(float(model(pooled).item()), 4)

            # Generate caption
            caption = ""
            try:
                generated_ids = state.whisper.generate(input_features)
                caption = state.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                caption = sanitize_caption(caption.strip())
            except Exception:
                pass

        elapsed = round(time.time() - t0, 3)

        return {
            "status": "ok",
            "scores": scores,
            "caption": caption,
            "elapsed": elapsed,
        }

    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    state.device = torch.device(f"cuda:{args.gpu}")
    print(f"EI server starting on GPU {args.gpu}, port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
