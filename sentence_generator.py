"""
Sentence generation via LFM 2.5 (VLLM OpenAI-compatible API).

Generates emotional and neutral sentences with specific constraints:
- Starting letter
- Word count range
- Punctuation requirements (exclamation marks, question marks, ellipsis)
- Emotion intensity matching the bucket
"""

import json
import random
import re
import string
import time
from typing import Optional

import requests

from config import (
    VLLM_PORT, WORD_COUNT_MIN, WORD_COUNT_MAX,
    ALL_DIMENSIONS, TOPICS_FILE,
    get_emotion_description,
)


def load_topics():
    """Load topics from topics.json."""
    with open(TOPICS_FILE) as f:
        return json.load(f)


TOPICS = None

def get_random_topic():
    global TOPICS
    if TOPICS is None:
        TOPICS = load_topics()
    return random.choice(TOPICS)


def sample_punctuation_params():
    """Sample punctuation parameters according to spec.

    Returns dict with:
        exclamation_count: number of ! to use
        question_count: number of ? to use
        use_ellipsis: whether to use "..."
    """
    # Exclamation marks: 33% zero, 33% 1-2, 34% 3+
    r = random.random()
    if r < 0.33:
        excl = 0
    elif r < 0.66:
        excl = random.randint(1, 2)
    else:
        excl = random.randint(3, 5)

    # Question marks: same distribution
    r = random.random()
    if r < 0.33:
        quest = 0
    elif r < 0.66:
        quest = random.randint(1, 2)
    else:
        quest = random.randint(3, 5)

    # Ellipsis: 50/50
    ellipsis = random.random() < 0.5

    return {
        "exclamation_count": excl,
        "question_count": quest,
        "use_ellipsis": ellipsis,
    }


def build_emotional_prompt(
    topic: str,
    letter: str,
    word_count: int,
    dimension: str,
    bucket: tuple,
    punctuation: dict,
) -> str:
    """Build a prompt for generating an emotional sentence."""
    emotion_desc = get_emotion_description(dimension, bucket)
    dim_info = ALL_DIMENSIONS.get(dimension, {})
    scale_info = dim_info.get("scale", "")

    # Punctuation instructions
    punct_parts = []
    if punctuation["exclamation_count"] == 0:
        punct_parts.append("Do NOT use any exclamation marks.")
    elif punctuation["exclamation_count"] <= 2:
        punct_parts.append(f"Use exactly {punctuation['exclamation_count']} exclamation mark(s).")
    else:
        punct_parts.append(f"Use {punctuation['exclamation_count']} or more exclamation marks for emphasis.")

    if punctuation["question_count"] == 0:
        punct_parts.append("Do NOT use any question marks.")
    elif punctuation["question_count"] <= 2:
        punct_parts.append(f"Use exactly {punctuation['question_count']} question mark(s).")
    else:
        punct_parts.append(f"Use {punctuation['question_count']} or more question marks.")

    if punctuation["use_ellipsis"]:
        punct_parts.append('Include "..." (ellipsis) somewhere as a thinking pause.')
    else:
        punct_parts.append('Do NOT use "..." (ellipsis).')

    punct_instruction = " ".join(punct_parts)

    system = "You are a voice actor script writer. Output ONLY the sentence, nothing else. No quotes, no labels, no explanation."

    user = (
        f"Write a single sentence about the topic: {topic}\n\n"
        f"Requirements:\n"
        f"- The sentence MUST begin with the capital letter '{letter}'\n"
        f"- The sentence must be approximately {word_count} words long\n"
        f"- The sentence must express '{dimension}' at a level described as: {emotion_desc}\n"
        f"  (Scale: {scale_info})\n"
        f"- Punctuation: {punct_instruction}\n"
        f"- Make it sound natural, like something a person would actually say\n"
        f"- Output ONLY the sentence, nothing else"
    )

    return system, user


def build_neutral_prompt(
    topic: str,
    letter: str,
    word_count: int,
) -> str:
    """Build a prompt for generating a boring/neutral sentence."""
    system = "You are a voice actor script writer. Output ONLY the sentence, nothing else. No quotes, no labels, no explanation."

    user = (
        f"Write a single, boring, emotionally flat sentence about the topic: {topic}\n\n"
        f"Requirements:\n"
        f"- The sentence MUST begin with the capital letter '{letter}'\n"
        f"- The sentence must be approximately {word_count} words long\n"
        f"- The sentence must be completely neutral and unemotional\n"
        f"- No exclamation marks, no question marks, no ellipsis\n"
        f"- It should sound like a dry factual statement\n"
        f"- Output ONLY the sentence, nothing else"
    )

    return system, user


GEMINI_API_KEY = "AIzaSyAKfhr11N5S9kXnUpgFEDDFW_L4iwrxNdg"

# LLM backend: "vllm" or "gemini"
LLM_BACKEND = "gemini"  # Default to gemini since VLLM may not be installed


def query_vllm(system_prompt: str, user_prompt: str, port: int = VLLM_PORT,
               max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Query VLLM server with chat completion format."""
    url = f"http://localhost:{port}/v1/chat/completions"

    payload = {
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\n"],
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def query_gemini(system_prompt: str, user_prompt: str,
                 max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Query Gemini API for sentence generation."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    # Gemini sometimes wraps in quotes or adds labels
    text = text.strip('"\'')
    if text.startswith("Sentence:"):
        text = text.split(":", 1)[1].strip()
    return text


def query_llm(system_prompt: str, user_prompt: str, port: int = VLLM_PORT,
              max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Query the configured LLM backend."""
    if LLM_BACKEND == "gemini":
        return query_gemini(system_prompt, user_prompt, max_tokens, temperature)
    else:
        return query_vllm(system_prompt, user_prompt, port, max_tokens, temperature)


def validate_sentence(sentence: str, letter: str, word_count: int,
                      tolerance: float = 0.4) -> tuple:
    """Validate a generated sentence.

    Returns (is_valid, issues_list).
    """
    issues = []

    # Clean up: remove quotes, labels
    sentence = sentence.strip('"\'')
    if sentence.startswith("Sentence:") or sentence.startswith("Output:"):
        sentence = sentence.split(":", 1)[1].strip()

    # Check starting letter
    if not sentence or sentence[0].upper() != letter.upper():
        issues.append(f"Does not start with '{letter}' (starts with '{sentence[0] if sentence else ''}')")

    # Check word count
    actual_words = len(sentence.split())
    min_ok = int(word_count * (1 - tolerance))
    max_ok = int(word_count * (1 + tolerance))
    if actual_words < min_ok or actual_words > max_ok:
        issues.append(f"Word count {actual_words} outside range [{min_ok}, {max_ok}] (target: {word_count})")

    return len(issues) == 0, issues, sentence


def generate_sentence(
    topic: str,
    letter: str,
    word_count: int,
    dimension: Optional[str] = None,
    bucket: Optional[tuple] = None,
    punctuation: Optional[dict] = None,
    is_emotional: bool = True,
    port: int = VLLM_PORT,
    max_retries: int = 3,
) -> dict:
    """Generate a sentence with validation and retry.

    Returns dict with:
        text: the generated sentence
        topic, letter, word_count_target, word_count_actual
        punctuation_params (if emotional)
        attempts: number of attempts needed
        valid: whether it passed validation
    """
    if is_emotional and punctuation is None:
        punctuation = sample_punctuation_params()

    best_sentence = None
    best_issues = None

    for attempt in range(max_retries):
        try:
            if is_emotional:
                system, user = build_emotional_prompt(
                    topic, letter, word_count, dimension, bucket, punctuation)
            else:
                system, user = build_neutral_prompt(topic, letter, word_count)

            raw = query_llm(system, user, port=port)
            valid, issues, cleaned = validate_sentence(raw, letter, word_count)

            if valid or best_sentence is None:
                best_sentence = cleaned
                best_issues = issues

            if valid:
                break

        except Exception as e:
            best_issues = [f"API error: {e}"]

    actual_words = len(best_sentence.split()) if best_sentence else 0

    result = {
        "text": best_sentence or f"{letter}nknown sentence generation failed.",
        "topic": topic,
        "letter": letter,
        "word_count_target": word_count,
        "word_count_actual": actual_words,
        "is_emotional": is_emotional,
        "attempts": min(attempt + 1, max_retries) if best_sentence else max_retries,
        "valid": best_issues is not None and len(best_issues) == 0,
        "issues": best_issues or [],
    }

    if is_emotional:
        result["punctuation_params"] = punctuation
        result["dimension"] = dimension
        result["bucket"] = list(bucket) if bucket else None

    return result


if __name__ == "__main__":
    # Test sentence generation
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--port", type=int, default=VLLM_PORT)
    parser.add_argument("--n", type=int, default=5, help="Number of test sentences")
    args = parser.parse_args()

    if args.test:
        topics = load_topics()
        print("Testing emotional sentence generation...")
        for i in range(args.n):
            topic = random.choice(topics)
            letter = random.choice(string.ascii_uppercase)
            word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)
            punct = sample_punctuation_params()

            result = generate_sentence(
                topic=topic, letter=letter, word_count=word_count,
                dimension="Anger", bucket=(3, 4), punctuation=punct,
                is_emotional=True, port=args.port,
            )
            print(f"\n[{i+1}] Topic: {topic}")
            print(f"    Letter: {letter}, Words: {word_count}")
            print(f"    Punct: {punct}")
            print(f"    Text: {result['text']}")
            print(f"    Valid: {result['valid']}, Actual words: {result['word_count_actual']}")
            if result['issues']:
                print(f"    Issues: {result['issues']}")

        print("\n\nTesting neutral sentence generation...")
        for i in range(args.n):
            topic = random.choice(topics)
            letter = random.choice(string.ascii_uppercase)
            word_count = random.randint(WORD_COUNT_MIN, WORD_COUNT_MAX)

            result = generate_sentence(
                topic=topic, letter=letter, word_count=word_count,
                is_emotional=False, port=args.port,
            )
            print(f"\n[{i+1}] Topic: {topic}")
            print(f"    Letter: {letter}, Words: {word_count}")
            print(f"    Text: {result['text']}")
            print(f"    Valid: {result['valid']}, Actual words: {result['word_count_actual']}")
