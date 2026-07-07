"""Corpus loader for Jacobian fitting.

Loads ~N prompts of >=128 chars from WikiText-103 + c4 (a pretraining-like
distribution). Truncates to max_seq_len tokens during fitting. Cached as
JSONL in data/corpus/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data" / "corpus"


def load_prompts(n: int = 200, min_chars: int = 200, *, force_refresh: bool = False) -> list[str]:
    """Load ~n prompts of >=min_chars from WikiText-103 + c4.

    Cached at data/corpus/prompts_{n}_{min_chars}.jsonl.
    """
    cache_path = CACHE_DIR / f"prompts_{n}_{min_chars}.jsonl"
    if cache_path.exists() and not force_refresh:
        with open(cache_path) as f:
            return [json.loads(line)["text"] for line in f]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []

    # Try WikiText-103 first (small, local-friendly).
    try:
        from datasets import load_dataset
        print(f"Loading WikiText-103...", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        for ex in ds:
            text = ex["text"].strip()
            if len(text) >= min_chars:
                prompts.append(text[:1000])  # cap length
                if len(prompts) >= n:
                    break
        print(f"  got {len(prompts)} prompts from WikiText-103", flush=True)
    except Exception as e:
        print(f"  WikiText-103 failed: {e}", flush=True)

    # If not enough, supplement with c4.
    if len(prompts) < n:
        try:
            from datasets import load_dataset
            need = n - len(prompts)
            print(f"Loading c4 (need {need} more)...", flush=True)
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            for ex in ds:
                text = ex["text"].strip()
                if len(text) >= min_chars:
                    prompts.append(text[:1000])
                    if len(prompts) >= n:
                        break
        except Exception as e:
            print(f"  c4 failed: {e}", flush=True)

    # If still not enough (offline), use a built-in fallback corpus.
    if len(prompts) < n:
        print(f"  using built-in fallback corpus ({n - len(prompts)} more needed)", flush=True)
        fallback = _FALLBACK_PROMPTS
        for p in fallback:
            if len(prompts) >= n:
                break
            if len(p) >= min_chars:
                prompts.append(p)

    # Cache
    with open(cache_path, "w") as f:
        for p in prompts[:n]:
            f.write(json.dumps({"text": p}) + "\n")
    return prompts[:n]


# A small built-in corpus for offline use / testing.
_FALLBACK_PROMPTS = [
    "The history of computing is a fascinating journey through human ingenuity. From the abacus to modern quantum computers, each era has built upon the innovations of the previous one. The invention of the transistor at Bell Labs in 1947 marked a pivotal moment, leading to integrated circuits and the microprocessor.",
    "Artificial intelligence has transformed how we interact with technology in the twenty first century. Machine learning models can now perform tasks that were once thought to require human intelligence, from image recognition to natural language understanding and generation.",
    "The ocean covers more than seventy percent of Earth's surface and contains the vast majority of the planet's water. It plays a crucial role in regulating global climate, absorbing carbon dioxide, and supporting diverse ecosystems that range from microscopic plankton to the largest whales.",
    "Mathematics is the language of science and engineering. From the ancient Greeks who developed geometry and logic, to modern researchers working on topology and number theory, mathematical thinking has shaped our understanding of the world and enabled countless technological advances.",
    "The human brain is a remarkable organ composed of approximately eighty six billion neurons connected by trillions of synapses. It is responsible for our thoughts, memories, emotions, and the control of all bodily functions, making it one of the most complex structures known to science.",
    "Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures, caused by greenhouse gas emissions from human activities, are leading to more frequent extreme weather events, rising sea levels, and disruptions to ecosystems and agriculture worldwide.",
    "The invention of the printing press by Johannes Gutenberg in the fifteenth century revolutionized the dissemination of knowledge in Europe. Before this innovation, books were copied by hand, making them expensive and rare. The printing press enabled mass production of texts.",
    "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This fundamental biological process produces oxygen as a byproduct and forms the base of most food chains on Earth, supporting nearly all life.",
    "The Roman Empire was one of the largest and most influential civilizations in world history. At its peak, it controlled territories spanning from Britain to Egypt and from Spain to the Middle East, leaving a lasting legacy in law, language, architecture, and governance.",
    "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the smallest scales. It reveals a world where particles can exist in multiple states simultaneously, where measurement affects the system being measured, and where entanglement connects distant particles.",
] * 30  # repeat to fill