# Retrieval-Augmented Voice Cloning & Emotion Conditioning Pipeline

A large-scale data generation pipeline that produces emotionally conditioned speech datasets with explicitly **disentangled speaker identity and emotional prosody**. By combining zero-shot TTS, voice conversion, and emotion scoring models, this pipeline generates controlled "triplets" where the speaker's voice and the emotion being expressed can be independently varied.

The resulting datasets can be used to train **omni-models** that accept natural-language emotion instructions (e.g., "say this more angrily" or "make this sound sadder") while preserving the target speaker's voice identity.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [How the Triplet System Works](#how-the-triplet-system-works)
- [Seed Variation and Best-of-N Selection](#seed-variation-and-best-of-n-selection)
- [Emotion Scoring with Empathic Insight Voice+](#emotion-scoring-with-empathic-insight-voice)
- [Speaker KV Scaling](#speaker-kv-scaling)
- [Echo TTS Sampling Guide](#echo-tts-sampling-guide)
- [Dataset Design for Omni-Model Training](#dataset-design-for-omni-model-training)
- [Published Dataset](#published-dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [GPU Requirements](#gpu-requirements)
- [Throughput & Performance](#throughput--performance)
- [Related Repositories](#related-repositories)
- [License](#license)

---

## Overview

The core insight behind this pipeline is **identity swapping**: for each highly emotional target audio clip, we generate a training triplet that explicitly separates "who is speaking" from "how they are speaking." This is achieved through a carefully orchestrated sequence of voice conversion, text-to-speech generation, and emotion filtering steps.

Each triplet consists of:

1. **Target**: The original, highly emotional audio clip from the source dataset
2. **Speaker Reference (Sample A)**: A neutral/boring sentence voice-converted to sound like the Target speaker — this provides the voice identity signal without emotional contamination
3. **Emotion Reference (Sample B)**: An emotional paraphrase of the Target's text, synthesized to match the Target's emotion, but voice-converted to sound like a *different* speaker — this provides the emotional prosody signal without the Target's identity

A model trained on these triplets learns that Sample A tells it *whose voice to use*, while Sample B tells it *what emotion to express* — because those two signals come from different speakers.

### Why This Matters

Standard TTS datasets conflate speaker identity and emotion: if Speaker X is always angry, the model learns "angry = Speaker X's voice." By swapping identities through voice conversion, we break that correlation. The model is forced to learn emotion as a transferable property, independent of any particular voice.

---

## Architecture

```
                              ┌─────────────────────────────┐
                              │   Emotion Reference Dataset  │
                              │  (HuggingFace, DACVAE-coded) │
                              └──────────┬──────────────────┘
                                         │
                        ┌────────────────┼────────────────┐
                        ▼                ▼                ▼
                  ┌──────────┐    ┌──────────┐    ┌──────────────┐
                  │  Target   │    │ Neutral   │    │  LLM (vLLM)  │
                  │  Audio    │    │  Sample   │    │  Paraphrase  │
                  └─────┬────┘    └─────┬────┘    └──────┬───────┘
                        │               │                │
                        │          ┌────▼────┐    ┌──────▼───────┐
                        │          │ Voice    │    │   Echo TTS    │
                        │          │ Convert  │    │  (N seeds)    │
                        │          │ → Target │    │  + Emotion    │
                        │          │ Identity │    │    Ref Audio  │
                        │          └────┬────┘    └──────┬───────┘
                        │               │                │
                        │               │          ┌─────▼──────┐
                        │               │          │  Empathic   │
                        │               │          │  Insight    │
                        │               │          │  Scoring    │
                        │               │          │  + Select   │
                        │               │          │  Best Seed  │
                        │               │          └─────┬──────┘
                        │               │                │
                        │               │          ┌─────▼──────┐
                        │               │          │ Voice       │
                        │               │          │ Convert     │
                        │               │          │ → Neutral   │
                        │               │          │ Identity    │
                        │               │          └─────┬──────┘
                        │               │                │
                        ▼               ▼                ▼
                  ┌──────────────────────────────────────────────┐
                  │              Training Triplet                 │
                  │                                               │
                  │  Target Audio    Speaker Ref    Emotion Ref   │
                  │  (emotional,     (neutral,      (emotional,   │
                  │   original       target voice)  neutral voice)│
                  │   speaker)                                    │
                  └──────────────────────────────────────────────┘
```

### Server-Based Processing

The pipeline runs as a set of FastAPI microservices, each hosting a different model on a dedicated GPU:

| Server | Model | Default Port | GPU | Purpose |
|--------|-------|-------------|-----|---------|
| `echo_tts_server.py` | Echo TTS (DiT + S1-DAC) | 920X | Configurable | Text-to-speech generation with speaker reference |
| `vc_server.py` | ChatterboxVC (S3Gen) | 930X | Configurable | Zero-shot voice conversion |
| `ei_server.py` | Empathic Insight Voice+ | 940X | Configurable | 59-expert emotion scoring & audio captioning |
| `vllm_server.py` | LFM 2.5 1.2B | 9100 | Shared | LLM-based sentence generation & paraphrasing |

Workers make HTTP requests to these servers, enabling easy scaling across multiple GPUs by running multiple sets of workers + servers.

---

## Key Components

### Echo TTS — Text-to-Speech

[Open Echo TTS](https://github.com/julien-blanchon/open-echo-tts) is a diffusion transformer (DiT) model that generates speech from text using a speaker reference audio. It works in the compressed latent space of the **Fish-Speech S1-DAC** codec ([jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min) on HuggingFace), which compresses 44.1kHz audio down to ~21.5 latent frames/second at 80 dimensions (after PCA). The model uses rectified flow with Classifier-Free Guidance (CFG) for both text and speaker conditioning.

Key parameters:
- **num_steps**: Number of Euler denoising steps (default: 40)
- **cfg_scale_text**: Text guidance strength (default: 3.0)
- **cfg_scale_speaker**: Speaker/emotion guidance strength (default: 8.0)
- **speaker_kv_scale**: Multiplier for speaker attention keys/values (see [Speaker KV Scaling](#speaker-kv-scaling))

**Important**: Echo TTS uses **S1-DAC** (44.1kHz, transformer-based quantizer) as its audio codec — not DACVAE. The DACVAE codec (Facebook, 48kHz) is only used for storing/loading audio samples in the source emotion-attribute dataset.

### ChatterboxVC — Voice Conversion

[ChatterboxVC](https://github.com/LAION-AI/chatterbox-voice-conversion) is a zero-shot voice conversion library built on Resemble AI's S3Gen model. Given a source audio and a target speaker reference, it produces audio with the source's content and prosody but the target's voice identity. Output sample rate is 24kHz.

The pipeline uses ChatterboxVC twice per triplet:
1. Convert neutral audio → Target speaker identity (creating Sample A)
2. Convert emotional TTS output → Neutral speaker identity (creating Sample B)

### Empathic Insight Voice+ — Emotion Scoring

[Empathic Insight Voice+](https://github.com/LAION-AI/emotion-annotations) is an ensemble of 59 expert models that score audio across 55 emotion dimensions and 4 quality metrics. It uses a fine-tuned Whisper encoder ([laion/BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper)) as the audio backbone, feeding embeddings into specialized MLP heads ([laion/Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)).

The 55 emotion dimensions include: Amusement, Anger, Fear, Sadness, Longing, Elation, Contentment, Disgust, Distress, Pain, Helplessness, Hope/Enthusiasm, Bitterness, Contempt, Doubt, and many more.

The 4 quality metrics: Overall Quality, Speech Quality, Content Enjoyment, Background Quality.

This model serves two critical roles in the pipeline:
1. **Source selection**: Ranking emotion reference clips by their target emotion score to find the most intensely emotional examples
2. **Output filtering**: Scoring all seed variations of TTS output and selecting the one that best matches the desired emotion

---

## How the Triplet System Works

### Step-by-Step for a Single Sample

Let's walk through generating one complete triplet targeting "Anger" at intensity level 3-4:

1. **Select Target**: Pull a highly angry audio clip from the emotion-attribute dataset (scored 3.0-4.0 on Anger by Empathic Insight)

2. **Select Neutral Sample**: Pick a boring/neutral sample — 50% chance from a random different emotion bucket, 50% from "Emotional Numbness" bucket

3. **First Voice Conversion** (Neutral → Target Identity):
   - Take the neutral audio
   - Use ChatterboxVC to clone the Target speaker's voice onto it
   - Result: **Sample A** — a neutral sentence in the Target's voice

4. **LLM Paraphrasing**:
   - Extract the Target's transcription
   - Use the LLM to write an emotionally similar but paraphrased sentence
   - Length varies: 30% shorter, 30% longer, 40% same length as original

5. **TTS Generation** (Multiple Seeds):
   - Use Echo TTS with the paraphrased text
   - Pass the Target audio as the speaker/emotion reference
   - Generate 3+ versions with different random seeds
   - Each seed produces a different prosodic realization

6. **Emotion Filtering** (Best-of-N Selection):
   - Score all seed variations with Empathic Insight
   - Select the variation with the highest Anger score
   - Discard the others

7. **Second Voice Conversion** (Emotional → Neutral Identity):
   - Take the winning TTS output
   - Voice-convert it to sound like the original Neutral speaker
   - Result: **Sample B** — an angry sentence in the Neutral speaker's voice

8. **Concatenation**: Sample A + 1s 10kHz sine separator + Sample B

---

## Seed Variation and Best-of-N Selection

One of the most powerful aspects of this pipeline is **seed-based diversity**. When Echo TTS generates speech, the random seed controls the initial noise tensor, which determines the specific prosodic realization: where pauses fall, how pitch contours flow, which syllables get emphasis, and whether the emotional delivery leans toward shouting vs. seething, sobbing vs. sighing, etc.

### Why Seeds Matter

For a given text + speaker reference combination, different seeds can produce dramatically different emotional scores:

```
Text: "Right now, the thought of all those unlived moments fills me with such dread..."
Emotion Reference: Fear (scored 4.0)

Seed 42:   Fear=0.96  (barely fearful)
Seed 137:  Fear=1.04  (slight fear)
Seed 512:  Fear=1.11  (slight fear)
Seed 3141: Fear=1.83  (moderate fear)   ← best of 10
Seed 7777: Fear=0.73  (barely fearful)
```

The same text and reference can produce fear scores ranging from 0.73 to 1.83 — a 2.5x difference! The best-of-N selection strategy picks seed 3141, significantly improving dataset quality compared to using a single random seed.

### Practical Recommendations

- **3 seeds** per sample: Good balance of quality vs. compute cost. Catches the worst outliers.
- **10 seeds** per sample: Substantially better quality. The best-of-10 score is typically 1.5-2x higher than the average score across all seeds.
- **More seeds**: Diminishing returns beyond ~10, but useful for high-stakes samples.

### Variety as a Feature, Not a Bug

The variation across seeds isn't just noise — it's genuinely different ways of expressing the same emotion. For dataset generation, you might intentionally keep multiple seed variations per sample:

- **Best seed**: Highest emotion score — the most intensely emotional version
- **Second-best seed**: Still emotional but with different prosodic choices — adds diversity to training data
- **Worst seed**: Can serve as a negative example or as the "less emotional" version in a pair

This natural variation is what enables training an omni-model with graded emotion instructions like "make this *slightly* more amused" vs. "make this *much* more amused."

---

## Emotion Scoring with Empathic Insight Voice+

### The 59-Expert Ensemble

Empathic Insight Voice+ ([laion/Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)) uses a fine-tuned Whisper encoder as its audio backbone. The Whisper embeddings are fed into 59 specialized MLP expert heads:

- **55 emotion experts** (FullEmbeddingMLP): Take the full sequence of Whisper embeddings and score specific emotions. Each expert was trained independently on human annotations for that specific emotion dimension.
- **4 quality experts** (PooledEmbeddingMLP): Use pooled (mean/min/max/std) embeddings to score overall audio quality, speech quality, content enjoyment, and background quality.

### Score Interpretation

Emotion scores are on a 0-4 scale:
| Score | Meaning |
|-------|---------|
| 0-1 | Not present or barely detectable |
| 1-2 | Slightly present |
| 2-3 | Moderately present |
| 3-4 | Strongly to extremely present |

### Why 40 Dimensions Matter

Having 40 emotion dimensions (not just "happy/sad/angry/neutral") allows for extremely fine-grained control. Many real emotional expressions are blends:

- **Bittersweet nostalgia**: High Longing + moderate Sadness + slight Contentment
- **Nervous excitement**: High Hope/Enthusiasm + moderate Fear + high Arousal
- **Sarcastic amusement**: High Amusement + moderate Contempt + low Warmth

The pipeline scores ALL 40 dimensions for every generated clip, so the dataset captures these nuanced blends, not just the primary target emotion. This is critical for training models that can handle complex emotional instructions.

### Audio Captioning

In addition to numerical scores, Empathic Insight generates natural-language captions describing the audio (e.g., "A male speaker with a warm, somewhat anxious tone speaks quickly about missed opportunities. Moderate background noise."). These captions can be used as additional training signal for instruction-following models.

---

## Speaker KV Scaling

### What It Does

The `speaker_kv_scale` parameter in Echo TTS directly multiplies the key and value vectors in the speaker attention mechanism. When set to 1.0 (default), speaker conditioning operates normally. Values above 1.0 amplify how strongly the model attends to the speaker/emotion reference audio.

This is a separate mechanism from Classifier-Free Guidance (CFG). CFG works by contrasting conditioned vs. unconditioned predictions at the output level. KV scaling works at the attention level, directly boosting the information flow from the reference audio through the transformer layers.

### Experimental Results

We conducted controlled experiments comparing baseline (no KV scaling), KV scale 1.1, and KV scale 1.5 across 5 emotions (Fear, Anger, Sadness, Amusement, Longing) with 5 emotion references each, 10 seeds per reference, using the same sentences and references across all runs:

| Emotion | Baseline | KV 1.1 | KV 1.5 | Winner |
|---------|----------|--------|--------|--------|
| **Fear** | **1.80** | 1.47 | 1.08 | Baseline |
| **Anger** | 1.80 | 1.73 | **2.53** | KV 1.5 |
| **Sadness** | 1.58 | **1.68** | 1.62 | KV 1.1 |
| **Amusement** | 1.67 | 1.82 | **3.35** | KV 1.5 |
| **Longing** | **3.09** | 2.89 | 1.92 | Baseline |
| **Overall Avg** | 1.99 | 1.92 | 2.10 | KV 1.5 |

*(Best-of-10 seeds, scored by Empathic Insight Voice+)*

### Key Findings

- **KV 1.5 is too aggressive for general use**: It dramatically boosts high-arousal emotions (Amusement doubled from 1.67 to 3.35, Anger increased 40%) but *destroys* low-arousal emotions (Fear dropped 40%, Longing dropped 38%)
- **KV 1.1 is a safe, subtle boost**: It provides a gentle 10% amplification that slightly improves some emotions (Sadness +6%, Amusement +9%) without breaking others. Fear still drops somewhat (-18%), suggesting even mild KV scaling shifts the model toward higher-arousal outputs.
- **Baseline is the safest general setting**: Most balanced across all emotion types, especially for subtle/low-arousal emotions like Fear and Longing
- **The pattern is clear**: KV scaling amplifies high-arousal emotions at the expense of low-arousal/subtle ones. There is no single KV scale that improves all emotions uniformly.

### Recommendations

- **For production pipelines**: Use `speaker_kv_scale=1.1` as a default. It provides a mild boost without significant degradation on any emotion.
- **For high-arousal emotions** (Anger, Amusement, Distress): `speaker_kv_scale=1.5` can produce significantly more emotionally intense output.
- **For low-arousal emotions** (Fear, Longing, Contentment, Sadness): Stick with baseline settings (no KV scaling).
- **For maximum flexibility**: Generate with multiple KV scales and let the Empathic Insight filtering pick the best result. The per-emotion score comparison will naturally select the optimal setting for each emotion type.

---

## Echo TTS Sampling Guide

A comprehensive sampling guide is included in [`echo_tts_sampling_guide.txt`](echo_tts_sampling_guide.txt). It covers:

- How the rectified flow diffusion process works
- The S1-DAC audio codec architecture (NOT DACVAE — see the guide for the distinction)
- Detailed explanation of every sampler parameter: `num_steps`, `cfg_scale_text`, `cfg_scale_speaker`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`, `speaker_kv_scale`, `speaker_kv_max_layers`, `speaker_kv_min_t`
- The PCA projection that compresses 1024-dim latents to 80-dim
- Practical tuning advice with examples

---

## Dataset Design for Omni-Model Training

### The Core Idea: Graded Emotion Pairs

The real power of this pipeline emerges when you generate *multiple versions* of the same content at different emotional intensities. By having one clip that scores 1.5 on Amusement and another that scores 3.5, you can create training pairs with instructions like:

- "Make this clip more amused" (1.5 → 3.5)
- "Make this clip less amused" (3.5 → 1.5)
- "Shift the emotion from sadness to anger" (high Sadness → high Anger)

### Triplet Structure for Training

Each training triplet provides three signals:

```
Concatenated Input:  [Speaker Ref (neutral, target voice)] + [Separator] + [Emotion Ref (emotional, different voice)]
                                    ↓                                              ↓
                            "Use this voice"                            "Express this emotion"
                                    ↓                                              ↓
                            ┌─────────────────────────────────────────────────────────┐
                            │                     Target Audio                        │
                            │           (emotional, target voice)                     │
                            └─────────────────────────────────────────────────────────┘
```

The model learns to:
1. **Clone the voice** from the Speaker Reference
2. **Transfer the emotion** from the Emotion Reference
3. **Combine both** into output that matches the Target

Because the voice and emotion come from *different speakers* in the input, the model cannot cheat by simply copying — it must truly understand and disentangle these two aspects of speech.

### Scaling to 40,000+ Samples

The pipeline targets ~1,000 samples per emotion bucket across 40 emotion dimensions:

- **40 emotion dimensions** × 4 intensity buckets each = up to 160 buckets
- **Top-2 buckets** per emotion (moderate + strong intensity) = ~80 active buckets
- **1,000 samples per bucket** = ~40,000-80,000 total training triplets

### Multi-Emotion and Cross-Emotion Pairs

Beyond single-emotion comparisons, the dataset naturally supports cross-emotion training:

- Same speaker, same text, but one version with high Anger and another with high Fear
- Same emotion intensity but different speakers (via the identity swapping)
- Same speaker and emotion but different seeds (prosodic variation)

These combinations enable training models that understand emotion as a continuous, multi-dimensional space rather than a fixed set of categories.

### WebDataset Format

The final dataset is packaged in WebDataset format (`.tar` shards) for efficient streaming during training:

```
shard-00000.tar
├── sample_001.target.wav          # Original emotional audio
├── sample_001.speaker_ref.wav     # Sample A (neutral, target voice)
├── sample_001.emotion_ref.wav     # Sample B (emotional, neutral voice)
├── sample_001.concat.wav          # A + separator + B
├── sample_001.target.dac.pth      # DACVAE latent for target
├── sample_001.speaker_ref.dac.pth # DACVAE latent for speaker ref
├── sample_001.emotion_ref.dac.pth # DACVAE latent for emotion ref
├── sample_001.concat.dac.pth      # DACVAE latent for concatenation
└── sample_001.metadata.json       # Scores, text, emotion labels, etc.
```

---

## Published Dataset

The first generation run produced **72,100 samples** across 43 emotion and attribute dimensions, available as a WebDataset on HuggingFace:

**[TTS-AGI/voice-acting-pipeline-output](https://huggingface.co/datasets/TTS-AGI/voice-acting-pipeline-output)**

### Summary

| Metric | Value |
|--------|-------|
| Total samples | 72,100 |
| Total tar shards | 1,691 |
| Total size | 575.3 GB |
| Completed buckets | 78 |
| Compute time | ~658 GPU-hours |
| Avg time per sample | 32.9 seconds |
| Audio per sample | 3 emotional + 3 neutral + 1 reference = 7 WAVs |
| Metadata per sample | 59-dimension EI scores + captions + generation params |

### Per-Dimension Breakdown

| Dimension | Samples | Buckets | Tars | Size (GB) |
|-----------|---------|---------|------|-----------|
| Affection | 1,600 | 0to1, 1to2, 2to3, 3to4, 4to5 | 77 | 31.2 |
| Amusement | 1,050 | 0to1, 1to2, 2to3, 3to4, 4to5 | 46 | 17.3 |
| Anger | 2,000 | 0to1, 1to2, 2to3, 3to4, 4to5, 5to6 | 46 | 5.1 |
| Astonishment/Surprise | 1,050 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 17.2 |
| Awe | 2,000 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 19.4 |
| Bitterness | 1,950 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 16.4 |
| Concentration | 2,000 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 18.5 |
| Confusion | 2,000 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 17.1 |
| Contemplation | 450 | 0to1, 1to2, 2to3, 3to4 | 44 | 18.4 |
| Contempt | 1,950 | 0to1, 1to2, 2to3, 3to4, 4to5 | 45 | 9.8 |
| Contentment | 2,000 | 0to1, 1to2, 2to3, 3to4 | 44 | 18.4 |
| Disappointment | 1,950 | 0to1, 1to2, 2to3, 3to4, 4to5 | 43 | 17.6 |
| Disgust | 1,950 | 0to1, 2to3, 3to4 | 41 | 5.3 |
| Distress | 2,000 | 1to2, 3to4, 4to5 | 42 | 17.2 |
| Doubt | 2,000 | 3to4, 4to5 | 40 | 16.3 |
| Elation | 2,000 | 4to5, 5to6 | 40 | 16.5 |
| Embarrassment | 2,000 | 0to1, 1to2, 2to3 | 42 | 15.4 |
| Emotional Numbness | 1,950 | 1to2, 2to3, 3to4 | 40 | 5.5 |
| Fatigue/Exhaustion | 2,000 | 3to4, 4to5 | 40 | 4.8 |
| Fear | 2,000 | 2to3, 3to4 | 40 | 16.5 |
| Helplessness | 1,850 | 2to3, 3to4 | 37 | 3.5 |
| Hope/Enthusiasm | 2,000 | 5to6, 6to7 | 40 | 16.7 |
| Impatience/Irritability | 2,000 | 3to4, 4to5 | 40 | 15.9 |
| Infatuation | 2,000 | 3to4, 4to5 | 40 | 17.0 |
| Interest | 2,000 | 2to3, 3to4 | 40 | 16.4 |
| Intoxication | 2,000 | 3to4, 4to5 | 40 | 4.1 |
| Longing | 2,000 | 2to3, 3to4 | 40 | 17.4 |
| Malevolence/Malice | 2,000 | 2to3, 3to4 | 40 | 4.7 |
| Pain | 1,950 | 4to5, 5to6 | 39 | 4.3 |
| Pleasure/Ecstasy | 2,000 | 2to3, 3to4 | 40 | 17.3 |
| Pride | 2,000 | 3to4, 4to5 | 40 | 16.9 |
| Relief | 1,300 | 4to5, 5to6 | 40 | 17.1 |
| Sadness | 2,000 | 3to4, 4to5 | 40 | 17.7 |
| Sexual Lust | 2,000 | 3to4, 4to5 | 40 | 17.9 |
| Shame | 2,000 | 4to5, 5to6 | 40 | 17.5 |
| Sourness | 2,000 | 2to3, 3to4 | 40 | 5.4 |
| Teasing | 2,000 | 2to3, 3to4 | 40 | 15.6 |
| Thankfulness/Gratitude | 1,100 | 3to4, 4to5 | 62 | 26.4 |
| Triumph | 2,000 | 3to4, 4to5 | 40 | 18.2 |
| **Total** | **72,100** | | **1,691** | **575.3** |

*Additional attribute dimensions (Age, Arousal, Authenticity, etc.) have exploratory 10-sample tars but were not run at scale.*

### Tar Shard Format

Each tar file is a WebDataset shard named `{dimension}_{bucket}_{random_id}.tar` containing ~10-50 samples:

```
Anger_4to5_1092434485.tar
├── Anger_4to5_000.emotional_seed765532.wav    # Emotional TTS, seed 1
├── Anger_4to5_000.emotional_seed588447.wav    # Emotional TTS, seed 2
├── Anger_4to5_000.emotional_seed983636.wav    # Emotional TTS, seed 3
├── Anger_4to5_000.neutral_seed765532.wav      # Neutral TTS, seed 1
├── Anger_4to5_000.neutral_seed588447.wav      # Neutral TTS, seed 2
├── Anger_4to5_000.neutral_seed983636.wav      # Neutral TTS, seed 3
├── Anger_4to5_000.ref_audio.wav               # Speaker reference
├── Anger_4to5_000.json                        # Metadata + 59-dim EI scores
├── Anger_4to5_001.emotional_seed...wav
└── ...
```

### Sample Metadata Schema

```json
{
  "sample_id": "Anger_4to5_042",
  "dimension": "Anger",
  "bucket": [4, 5],
  "voice_conversion": { "used_vc": true, "laion_voice": "speaker_name.wav" },
  "emotional_sentence": { "text": "How dare they...", "topic": "injustice", "word_count_actual": 35 },
  "neutral_sentence": { "text": "The report discusses...", "topic": "statistics", "word_count_actual": 28 },
  "emotional_generations": [
    {
      "seed": 765532,
      "duration": 8.2,
      "ei_scores": { "Anger": 3.45, "Fear": 0.12, "Sadness": 0.34, "...": "..." },
      "caption": "An angry male voice speaks forcefully about perceived injustice..."
    }
  ],
  "neutral_generations": [ "..." ]
}
```

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU(s) with 16+ GB VRAM each
- PyTorch 2.0+

### Setup

```bash
# Clone this repository
git clone https://github.com/LAION-AI/Retrieval-Augmented-Voice-Cloning.git
cd Retrieval-Augmented-Voice-Cloning

# Install Python dependencies
pip install -r requirements.txt

# Install Echo TTS
pip install git+https://github.com/julien-blanchon/open-echo-tts.git

# Install DACVAE (for dataset I/O)
pip install git+https://github.com/kadirnar/fast-dacvae.git

# Install ChatterboxVC (requires separate Python 3.13 environment)
# See https://github.com/LAION-AI/chatterbox-voice-conversion for setup
pip install chatterbox-tts  # In a separate venv with Python 3.13

# Install vLLM for sentence generation
pip install vllm openai

# Run the automated installer for additional setup
bash install.sh
```

### Environment Notes

- **LD_LIBRARY_PATH**: If you're using conda environments, you may need to clear `LD_LIBRARY_PATH` to avoid cuDNN conflicts. The server scripts handle this automatically.
- **ChatterboxVC**: Requires a separate Python 3.13 virtual environment (`spiritvenv`) due to dependency conflicts. The VC server manages this via subprocess.
- **Audio I/O**: Use `soundfile` for WAV I/O — `torchcodec`/`torchaudio.load` may have FFmpeg compatibility issues on some systems. `torchaudio.functional.resample()` works fine for sample rate conversion.

---

## Configuration

All configuration is centralized in `config.py`:

```python
# GPU allocation
GPUS = [5, 6, 7]  # Which GPUs to use

# HuggingFace repositories
HF_UPLOAD_REPO = "TTS-AGI/voice-acting-pipeline-output"
EMOTION_REF_DATASET = "TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave"
LAION_REF_VOICES = "laion/clustered-reference-voices"

# Pipeline parameters
SAMPLES_PER_BUCKET = 10       # Samples per emotion bucket
SEEDS_PER_SAMPLE = 3          # Random seeds per TTS generation
ECHO_TTS_STEPS = 40           # Euler denoising steps
WORD_COUNT_MIN = 10           # Min words in generated sentences
WORD_COUNT_MAX = 70           # Max words in generated sentences

# Sample rates (these are fixed by the models)
DACVAE_SR = 48000             # DACVAE codec
ECHO_TTS_SR = 44100           # Echo TTS / S1-DAC codec
CHATTERBOX_SR = 24000         # ChatterboxVC output
EI_SR = 16000                 # Empathic Insight input
```

### Port Scheme

Each GPU gets its own set of server ports:
- **vLLM**: Port 9100 (shared)
- **Echo TTS** on GPU N: Port 9200+N
- **ChatterboxVC** on GPU N: Port 9300+N
- **Empathic Insight** on GPU N: Port 9400+N

---

## Usage

### Quick Test (5 samples)

```bash
# Start servers on GPU 5
LD_LIBRARY_PATH="" python servers/echo_tts_server.py --gpu 5 --port 9205 &
LD_LIBRARY_PATH="" python servers/vc_server.py --gpu 6 --port 9306 &
LD_LIBRARY_PATH="" python servers/ei_server.py --gpu 7 --port 9407 &

# Run test pipeline
LD_LIBRARY_PATH="" python test_pipeline.py --gpu 5 --samples 5 --dimension Anger --bucket 3to4
```

### Full Pipeline (Multi-GPU)

```bash
# Launch everything with the master orchestrator
LD_LIBRARY_PATH="" python run_pipeline.py --gpus 5,6,7

# Or for scaled production (1000 samples/bucket):
LD_LIBRARY_PATH="" python run_scaled.py --gpus 5,6,7
```

### Emotion Demos

```bash
# Generate emotion comparison HTML with 5 emotions × 5 refs × 10 seeds
LD_LIBRARY_PATH="" python demo_emotions_sametext.py

# With KV scaling = 1.1
LD_LIBRARY_PATH="" python demo_emotions_sametext_kv11.py
```

---

## GPU Requirements

### Minimum: Single GPU (16+ GB)

All three models can run sequentially on a single GPU with model offloading:
- Echo TTS: ~10-12 GB peak during inference
- ChatterboxVC: ~3-4 GB peak
- Empathic Insight: ~9-10 GB peak

### Recommended: 3 GPUs (16+ GB each)

One model per GPU for maximum throughput:
- GPU A: Echo TTS (~10-12 GB)
- GPU B: ChatterboxVC (~3-4 GB)
- GPU C: Empathic Insight (~9-10 GB)

### Production: 3+ GPU sets

Each GPU triplet runs an independent worker. N triplets = N× throughput.

---

## Throughput & Performance

Measured on NVIDIA A100-SXM4-80GB:

| Operation | Time per Sample |
|-----------|----------------|
| DACVAE decode (reference audio) | ~0.5s |
| Voice Conversion (ChatterboxVC) | ~1.6s |
| TTS Generation (Echo TTS, 40 steps) | ~4.5s |
| Emotion Scoring (Empathic Insight) | ~0.7s |
| **Full pipeline per sample** | **~8s** |
| **Full triplet (3 seeds + scoring)** | **~31s** |

**Throughput**: ~115 triplets/hour on a single A100 GPU set (3 GPUs)

---

## Related Repositories

| Repository | Description |
|-----------|-------------|
| [LAION-AI/chatterbox-voice-conversion](https://github.com/LAION-AI/chatterbox-voice-conversion) | Zero-shot voice conversion using Resemble AI's Chatterbox S3Gen model |
| [LAION-AI/emotion-annotations](https://github.com/LAION-AI/emotion-annotations) | Empathic Insight Voice+ emotion scoring model (59 experts) |
| [LAION-AI/Emo-Net-Voice](https://github.com/LAION-AI/Emo-Net-Voice) | Emotion network for voice analysis |
| [LAION-AI/emotional-speech-annotations](https://github.com/LAION-AI/emotional-speech-annotations) | Prompts and best practices for audio emotion annotation with ALMs |
| [LAION-AI/Vocalino-V0.1-Voice-Acting-Pipeline](https://github.com/LAION-AI/Vocalino-V0.1-Voice-Acting-Pipeline) | Earlier version: voice acting pipeline with natural-language direction |
| [julien-blanchon/open-echo-tts](https://github.com/julien-blanchon/open-echo-tts) | Open Echo TTS — diffusion transformer TTS model |
| [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) | Echo TTS model weights (HuggingFace) |
| [jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min) | S1-DAC audio codec weights used by Echo TTS |
| [kadirnar/fast-dacvae](https://github.com/kadirnar/fast-dacvae) | Fast DACVAE implementation for dataset audio encoding |
| [laion/BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) | Fine-tuned Whisper encoder used by Empathic Insight |
| [laion/Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) | 59 expert MLP heads for emotion scoring |
| [TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave](https://huggingface.co/datasets/TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets-DACVAE-Wave) | Source emotion-attribute dataset (DACVAE-encoded) |
| [laion/clustered-reference-voices](https://huggingface.co/datasets/laion/clustered-reference-voices) | 3000+ clustered LAION reference voices for identity diversity |

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Individual model components may have their own licenses:
- Echo TTS: Apache 2.0
- ChatterboxVC/S3Gen: Check [Resemble AI's license](https://github.com/resemble-ai/chatterbox)
- Empathic Insight: Check [LAION's license](https://github.com/LAION-AI/emotion-annotations)
