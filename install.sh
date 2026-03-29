#!/bin/bash
# Voice Acting Pipeline - Installation Script
# Run from the repository root directory

set -e

echo "=== Voice Acting Pipeline Setup ==="

# 1. Install Python dependencies
echo ""
echo "--- Installing Python dependencies ---"
pip install fastapi uvicorn python-multipart requests aiohttp soundfile librosa scipy

# 2. Install Echo TTS
echo ""
echo "--- Checking Echo TTS ---"
if python -c "from open_echo_tts.pipeline.loader import load_model" 2>/dev/null; then
    echo "Echo TTS already installed"
else
    echo "Installing Open Echo TTS..."
    pip install git+https://github.com/julien-blanchon/open-echo-tts.git
fi

# 3. Install VLLM (if not present)
echo ""
echo "--- Checking VLLM ---"
if python -c "import vllm" 2>/dev/null; then
    echo "VLLM already installed"
else
    echo "Installing VLLM..."
    pip install vllm openai
fi

# 4. Install DACVAE (if not present)
echo ""
echo "--- Checking DACVAE ---"
if python -c "from dacvae import DACVAE" 2>/dev/null; then
    echo "DACVAE already installed"
else
    echo "Installing fast-dacvae..."
    pip install git+https://github.com/kadirnar/fast-dacvae.git
fi

# 5. Create necessary directories
echo ""
echo "--- Creating directories ---"
mkdir -p tmp progress models_cache ID-refs

# 6. Verify Echo TTS can be imported
echo ""
echo "--- Verifying Echo TTS ---"
ECHO_TTS_SRC="${ECHO_TTS_SRC:-$(python -c 'import open_echo_tts; import os; print(os.path.dirname(os.path.dirname(open_echo_tts.__file__)))' 2>/dev/null || echo '')}"
if [ -n "$ECHO_TTS_SRC" ]; then
    echo "Echo TTS source: $ECHO_TTS_SRC"
else
    echo "WARNING: Echo TTS not found. Set ECHO_TTS_SRC environment variable."
fi

# 7. ChatterboxVC note
echo ""
echo "--- ChatterboxVC ---"
echo "ChatterboxVC requires a separate Python 3.13 environment."
echo "See: https://github.com/LAION-AI/chatterbox-voice-conversion"
echo "Set SPIRITVENV_PYTHON to point to the venv's python binary."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Place a speaker reference audio in ID-refs/speaker_ref.mp3"
echo "  2. Set environment variables (optional):"
echo "     export ECHO_TTS_SRC=~/open-echo-tts/src"
echo "     export SPIRITVENV_PYTHON=~/spiritvenv/bin/python"
echo "     export SPEAKER_REF=./ID-refs/speaker_ref.mp3"
echo "  3. Start servers: LD_LIBRARY_PATH='' python servers/echo_tts_server.py --gpu 0 --port 9200"
echo "  4. Test: python test_pipeline.py --gpu 0 --samples 1"
echo "  5. Full run: python master.py --gpus 0,1,2"
