"""
Central configuration for the Voice Acting Pipeline.

All constants, paths, dimension definitions, bucket ranges, score scales,
port schemes, and GPU allocation are defined here.
"""

import os

# ─── GPU Configuration ──────────────────────────────────────────────────────
# Which GPUs to use. Each GPU runs Echo TTS + VC + EI. VLLM is shared.
GPUS = [5, 6, 7]

# ─── HuggingFace Repositories ───────────────────────────────────────────────
HF_UPLOAD_REPO = "TTS-AGI/voice-acting-pipeline-output"
EMOTION_REF_DATASET = "TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave"
LAION_REF_VOICES = "laion/clustered-reference-voices"
LFM_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# ─── Pipeline Parameters ────────────────────────────────────────────────────
SAMPLES_PER_BUCKET = 10
SEEDS_PER_SAMPLE = 3
ECHO_TTS_STEPS = 40
SPEAKER_REF_MIN_DURATION = 6.0   # seconds
SPEAKER_REF_MAX_DURATION = 15.0  # seconds
WORD_COUNT_MIN = 10
WORD_COUNT_MAX = 70

# ─── Port Scheme ────────────────────────────────────────────────────────────
# VLLM: 9100 (shared, on first GPU)
# Echo TTS on GPU N: 9200+N
# ChatterboxVC on GPU N: 9300+N
# Empathic Insight on GPU N: 9400+N
VLLM_PORT = 9100
def echo_tts_port(gpu: int) -> int: return 9200 + gpu
def vc_port(gpu: int) -> int: return 9300 + gpu
def ei_port(gpu: int) -> int: return 9400 + gpu

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_FILE = os.path.join(BASE_DIR, "topics.json")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
LAION_VOICES_DIR = os.path.join(BASE_DIR, "laion_ref_voices")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")

ECHO_TTS_SRC = os.environ.get("ECHO_TTS_SRC", os.path.expanduser("~/open-echo-tts/src"))
SPIRITVENV_PYTHON = os.environ.get("SPIRITVENV_PYTHON", os.path.expanduser("~/spiritvenv/bin/python"))
DACVAE_WEIGHTS = os.environ.get("DACVAE_WEIGHTS", os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--dacvae-watermarked/snapshots/latest/weights.pth"))
EI_MODELS_CACHE = os.path.join(BASE_DIR, "models_cache")

# ─── Sample Rates ────────────────────────────────────────────────────────────
DACVAE_SR = 48000
ECHO_TTS_SR = 44100
CHATTERBOX_SR = 24000
EI_SR = 16000

# ─── Empathic Insight Config ────────────────────────────────────────────────
WHISPER_MODEL_ID = "laion/BUD-E-Whisper"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Plus"
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]
POOLED_DIM = 3072  # 4 * 768 (mean + min + max + std)
QUALITY_EXPERT_FILES = {
    "model_score_overall_quality_best.pth",
    "model_score_speech_quality_best.pth",
    "model_score_background_quality_best.pth",
    "model_score_content_enjoyment_best.pth",
}

# ─── Score Keys (55 emotion/attribute + 4 quality) ──────────────────────────
EMOTION_KEYS = [
    "Amusement", "Elation", "Pleasure_Ecstasy", "Contentment",
    "Thankfulness_Gratitude", "Affection", "Infatuation",
    "Hope_Enthusiasm_Optimism", "Triumph", "Pride", "Interest", "Awe",
    "Astonishment_Surprise", "Concentration", "Contemplation", "Relief",
    "Longing", "Teasing", "Impatience_and_Irritability", "Sexual_Lust",
    "Doubt", "Fear", "Distress", "Confusion", "Embarrassment", "Shame",
    "Disappointment", "Sadness", "Bitterness", "Contempt", "Disgust",
    "Anger", "Malevolence_Malice", "Sourness", "Pain", "Helplessness",
    "Fatigue_Exhaustion", "Emotional_Numbness",
    "Intoxication_Altered_States_of_Consciousness", "Jealousy_&_Envy",
]

ATTRIBUTE_KEYS = [
    "Valence", "Arousal", "Submissive_vs._Dominant", "Age", "Gender",
    "Serious_vs._Humorous", "Vulnerable_vs._Emotionally_Detached",
    "Confident_vs._Hesitant", "Warm_vs._Cold", "Monotone_vs._Expressive",
    "High-Pitched_vs._Low-Pitched", "Soft_vs._Harsh", "Authenticity",
    # NOTE: Recording_Quality and Background_Noise excluded - not voice dimensions
]

ALL_EI_KEYS = EMOTION_KEYS + ATTRIBUTE_KEYS

QUALITY_KEYS = [
    "score_overall_quality", "score_speech_quality",
    "score_content_enjoyment", "score_background_quality",
]

ALL_SCORE_KEYS = ALL_EI_KEYS + QUALITY_KEYS

# ─── Dimension Definitions ──────────────────────────────────────────────────
# Each dimension has a list of bucket ranges [min, max) and a description
# of what the scores mean at different levels.

# Emotions (40): 0-4 scale, softmax probability
# 0=not present, 1=slight, 2=moderate, 3=strong, 4=extremely present
# Bucket width = 1.0

EMOTION_DIMENSIONS = {}
for emo in EMOTION_KEYS:
    EMOTION_DIMENSIONS[emo] = {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0-4 (softmax probability)",
        "levels": {
            (0, 1): "not present or barely detectable",
            (1, 2): "slightly present",
            (2, 3): "moderately present",
            (3, 4): "strongly to extremely present",
        },
    }

# Attributes (15): varying scales
ATTRIBUTE_DIMENSIONS = {
    "Valence": {
        "buckets": [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)],
        "scale": "-3 (extremely negative) to +3 (extremely positive), 0=neutral",
        "levels": {
            (-3, -2): "extremely negative emotional valence",
            (-2, -1): "quite negative emotional valence",
            (-1, 0): "slightly negative emotional valence",
            (0, 1): "slightly positive emotional valence",
            (1, 2): "quite positive emotional valence",
            (2, 3): "extremely positive emotional valence",
        },
    },
    "Arousal": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0 (very calm) to 4 (very excited), 2=neutral",
        "levels": {
            (0, 1): "very calm and subdued energy",
            (1, 2): "slightly calm, low energy",
            (2, 3): "moderately excited, above neutral energy",
            (3, 4): "very excited, high energy",
        },
    },
    "Submissive_vs._Dominant": {
        "buckets": [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)],
        "scale": "-3 (very submissive) to +3 (very dominant), 0=neutral",
        "levels": {
            (-3, -2): "very submissive, meek",
            (-2, -1): "somewhat submissive",
            (-1, 0): "slightly submissive",
            (0, 1): "slightly dominant",
            (1, 2): "somewhat dominant, assertive",
            (2, 3): "very dominant, commanding",
        },
    },
    "Age": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        "scale": "0 (infant) to 6 (very old), 2=teenager, 4=adult",
        "levels": {
            (0, 1): "infant to very young child",
            (1, 2): "child",
            (2, 3): "teenager to young adult",
            (3, 4): "adult",
            (4, 5): "middle-aged to older",
            (5, 6): "elderly",
        },
    },
    "Gender": {
        "buckets": [(-2, -1), (-1, 0), (0, 1), (1, 2)],
        "scale": "-2 (very masculine) to +2 (very feminine), 0=neutral",
        "levels": {
            (-2, -1): "very masculine vocal characteristics",
            (-1, 0): "somewhat masculine",
            (0, 1): "somewhat feminine",
            (1, 2): "very feminine vocal characteristics",
        },
    },
    "Serious_vs._Humorous": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0 (very serious) to 4 (very humorous), 2=neutral",
        "levels": {
            (0, 1): "very serious tone",
            (1, 2): "somewhat serious",
            (2, 3): "somewhat humorous, light",
            (3, 4): "very humorous, comedic",
        },
    },
    "Vulnerable_vs._Emotionally_Detached": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0-4, 2=neutral",
        "levels": {
            (0, 1): "very vulnerable, emotionally exposed",
            (1, 2): "somewhat vulnerable",
            (2, 3): "somewhat emotionally detached",
            (3, 4): "very emotionally detached, stoic",
        },
    },
    "Confident_vs._Hesitant": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0-4, 2=neutral",
        "levels": {
            (0, 1): "very confident, certain",
            (1, 2): "somewhat confident",
            (2, 3): "somewhat hesitant, uncertain",
            (3, 4): "very hesitant, doubtful",
        },
    },
    "Warm_vs._Cold": {
        "buckets": [(-2, -1), (-1, 0), (0, 1), (1, 2)],
        "scale": "-2 (very cold) to +2 (very warm), 0=neutral",
        "levels": {
            (-2, -1): "very cold, distant tone",
            (-1, 0): "somewhat cold",
            (0, 1): "somewhat warm",
            (1, 2): "very warm, inviting tone",
        },
    },
    "Monotone_vs._Expressive": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0-4, 2=neutral",
        "levels": {
            (0, 1): "very monotone, flat delivery",
            (1, 2): "somewhat monotone",
            (2, 3): "somewhat expressive",
            (3, 4): "very expressive, dynamic delivery",
        },
    },
    "High-Pitched_vs._Low-Pitched": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0-4, 2=neutral",
        "levels": {
            (0, 1): "very high-pitched voice",
            (1, 2): "somewhat high-pitched",
            (2, 3): "somewhat low-pitched",
            (3, 4): "very low-pitched, deep voice",
        },
    },
    "Soft_vs._Harsh": {
        "buckets": [(-2, -1), (-1, 0), (0, 1), (1, 2)],
        "scale": "-2 (very harsh) to +2 (very soft), 0=neutral",
        "levels": {
            (-2, -1): "very harsh, rough vocal quality",
            (-1, 0): "somewhat harsh",
            (0, 1): "somewhat soft",
            (1, 2): "very soft, gentle vocal quality",
        },
    },
    "Authenticity": {
        "buckets": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "scale": "0 (artificial) to 4 (genuine), 2=neutral",
        "levels": {
            (0, 1): "very artificial sounding",
            (1, 2): "somewhat artificial",
            (2, 3): "somewhat genuine",
            (3, 4): "very genuine, authentic sounding",
        },
    },
    # NOTE: Recording_Quality and Background_Noise removed - not useful dimensions
    # to generate training data for (they describe recording conditions, not voice).
}

# Dimensions where voice conversion should NEVER be applied.
# Pitch and Age are intrinsic voice properties that VC would destroy.
NO_VC_DIMENSIONS = {"High-Pitched_vs._Low-Pitched", "Age"}

# Combined: all dimensions with their bucket info
ALL_DIMENSIONS = {}
ALL_DIMENSIONS.update(EMOTION_DIMENSIONS)
ALL_DIMENSIONS.update(ATTRIBUTE_DIMENSIONS)


def bucket_to_tar_name(dimension: str, bucket: tuple) -> str:
    """Convert dimension + bucket tuple to tar filename in HF dataset.
    E.g. ('Anger', (2, 3)) -> 'Anger_2to3.tar'
    """
    bmin, bmax = bucket
    # Handle negative values
    bmin_str = str(bmin).replace("-", "neg")
    bmax_str = str(bmax).replace("-", "neg")
    return f"{dimension}_{bmin_str}to{bmax_str}.tar"


def bucket_to_str(bucket: tuple) -> str:
    """Convert bucket tuple to string for filenames. E.g. (2,3) -> '2to3'"""
    bmin, bmax = bucket
    bmin_str = str(bmin).replace("-", "neg")
    bmax_str = str(bmax).replace("-", "neg")
    return f"{bmin_str}to{bmax_str}"


def get_emotion_description(dimension: str, bucket: tuple) -> str:
    """Get a human-readable description of what this bucket means."""
    dim_info = ALL_DIMENSIONS.get(dimension, {})
    levels = dim_info.get("levels", {})
    desc = levels.get(bucket, "")
    if not desc:
        # Generate generic description based on score range
        bmin, bmax = bucket
        if bmax <= 1:
            intensity = "not present or barely detectable"
        elif bmax <= 2:
            intensity = "slightly present"
        elif bmax <= 3:
            intensity = "moderately present"
        elif bmax <= 4:
            intensity = "strongly present"
        else:
            intensity = "extremely/intensely present"
        desc = f"{dimension.replace('_', ' ')} - {intensity} (score {bmin}-{bmax})"
    return desc


# ─── Dynamic Tar Discovery ──────────────────────────────────────────────────
# The actual dataset has tars that don't always match our hardcoded buckets.
# This function discovers what's actually available.

_available_tars_cache = None

def discover_available_tars():
    """Discover all available tar files from the HF emotion reference dataset.

    Returns dict: {dimension: [bucket_str, ...]}
    E.g. {'Anger': ['0to1', '2to3', '3to4', '4to5', '5to6'], ...}
    """
    global _available_tars_cache
    if _available_tars_cache is not None:
        return _available_tars_cache

    from huggingface_hub import HfApi
    api = HfApi()
    files = list(api.list_repo_tree(
        EMOTION_REF_DATASET, repo_type='dataset', path_in_repo='data'))

    from collections import defaultdict
    result = defaultdict(list)

    for f in files:
        path = f.path if hasattr(f, 'path') else str(f)
        if not path.endswith('.tar'):
            continue
        name = path.replace('data/', '').replace('.tar', '')

        # Parse: find the last part that looks like a bucket range
        # Format is either 'Dim_NtoM' or 'Dim_N.NN_to_M.MM'
        if '_to_' in name:
            # Float bucket: e.g. 'Age_0.00_to_0.86'
            # Skip these for now - we use integer buckets
            continue

        # Integer bucket: find the 'NtoM' at the end
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and 'to' in parts[1]:
            dim = parts[0]
            bucket_str = parts[1]
            result[dim].append(bucket_str)

    _available_tars_cache = dict(result)
    return _available_tars_cache


def get_available_buckets(dimension):
    """Get list of available bucket tuples for a dimension."""
    tars = discover_available_tars()
    bucket_strs = tars.get(dimension, [])
    buckets = []
    for bs in bucket_strs:
        parts = bs.split('to')
        try:
            bmin = int(parts[0])
            bmax = int(parts[1])
            buckets.append((bmin, bmax))
        except ValueError:
            continue
    return sorted(buckets)
