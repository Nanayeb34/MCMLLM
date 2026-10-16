"""AfriMed-VQA-Parallel loading and condition construction.

The dataset (gated; request access at
https://huggingface.co/datasets/afrimedqa/afrimed-vqa-parallel) ships as one
HF config per language, each a `test` split of ~300 four-way MCQs with the
image embedded as a base64 PNG string in the `image` column. All 14 configs
share the same underlying question/image pool translated in parallel, which
is what makes the English vs. low-resource comparison apples-to-apples.

Blind and Shuffled are both derived from AfriMed-VQA-Parallel itself, varying only in
what image (if any) is attached to the same fixed set of questions.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

from datasets import load_dataset
from PIL import Image

DATASET_REPO = "afrimedqa/afrimed-vqa-parallel"

# High-resource trio vs. the 11 low-resource languages, per the experimental
# design. Together these are exactly the dataset's 14 HF configs.
HIGH_RESOURCE: tuple[str, ...] = ("english", "french", "portuguese")
LOW_RESOURCE: tuple[str, ...] = (
    "amharic", "arabic", "hausa", "igbo", "isizulu", "sesotho",
    "swahili", "twi", "wolof", "xhosa", "yoruba",
)
LANGUAGES: tuple[str, ...] = HIGH_RESOURCE + LOW_RESOURCE

CONDITIONS: tuple[str, ...] = ("multimodal", "blind", "shuffled")
OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def resource_group(lang: str) -> str:
    return "high" if lang in HIGH_RESOURCE else "low"


@dataclass
class VQAItem:
    sample_id: int
    lang: str
    question: str
    options: dict[str, str]   # {"A": ..., "B": ..., "C": ..., "D": ...}
    correct_option: str       # one of OPTION_LETTERS
    image_b64: str            # raw base64 PNG, decoded lazily by the caller


def _row_to_item(row: dict, lang: str) -> VQAItem | None:
    options = {k: (row.get(k) or "").strip() for k in OPTION_LETTERS}
    correct = (row.get("correct_option") or "").strip().upper()
    # Each language config mixes MCQ rows (4 options + a correct_option
    # letter) with SAQ (short-answer, free-text) rows that have no options
    # at all — only the former fit a Δvisual accuracy comparison. A handful
    # of MCQ rows also have a malformed answer key; drop those too rather
    # than silently scoring them wrong.
    if correct not in OPTION_LETTERS or any(not v for v in options.values()):
        return None
    return VQAItem(
        sample_id=int(row["sample_id"]),
        lang=lang,
        question=row["question"],
        options=options,
        correct_option=correct,
        image_b64=row["image"],
    )


def load_language(lang: str) -> list[VQAItem]:
    """Load one language config's `test` split as a list of VQAItem."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language '{lang}'. Choose from: {LANGUAGES}")
    ds = load_dataset(DATASET_REPO, lang, split="test")
    rows = (_row_to_item(row, lang) for row in ds)
    return [r for r in rows if r is not None]


def decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _shuffle_partner(items: list[VQAItem], index: int, seed: int) -> int:
    """Deterministically map (item, seed) -> a *different* item's index, so
    the mismatched-image control is reproducible across reruns/resumption
    without storing the pairing anywhere.
    """
    n = len(items)
    digest = hashlib.sha256(f"{items[index].sample_id}:{seed}".encode()).hexdigest()
    offset = 1 + int(digest, 16) % (n - 1)
    return (index + offset) % n


@dataclass
class EvalInput:
    """One (item, condition) pair, ready to hand to a model."""
    sample_id: int
    lang: str
    condition: str
    question: str
    options: dict[str, str]
    correct_option: str
    image: Image.Image | None   # None only for the blind condition


def build_eval_inputs(items: list[VQAItem], condition: str, seed: int = 42) -> list[EvalInput]:
    """Turn one language's items into model-ready inputs for one condition.

    - multimodal: the item's own (correct) image.
    - blind:      no image — pure text + MCQ options.
    - shuffled:   another item's image from the same language pool, chosen
                  deterministically from `seed` (mismatched-case control).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from: {CONDITIONS}")

    out = []
    for i, it in enumerate(items):
        if condition == "multimodal":
            image = decode_image(it.image_b64)
        elif condition == "blind":
            image = None
        else:
            partner = _shuffle_partner(items, i, seed)
            image = decode_image(items[partner].image_b64)

        out.append(EvalInput(
            sample_id=it.sample_id,
            lang=it.lang,
            condition=condition,
            question=it.question,
            options=it.options,
            correct_option=it.correct_option,
            image=image,
        ))
    return out
