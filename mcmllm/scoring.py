"""Parsing model output into a predicted option letter, and scoring."""

from __future__ import annotations

import re

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(generated_text: str) -> str | None:
    """Extract the first standalone A/B/C/D token from the model's output.

    Returns None if no option letter could be found (counted as incorrect,
    but kept distinct from a wrong-but-parsed answer so parse failures are
    visible in aggregation rather than silently inflating error counts).
    """
    match = _LETTER_RE.search(generated_text.strip().upper())
    return match.group(1) if match else None


def is_correct(generated_text: str, correct_option: str) -> bool:
    predicted = parse_answer(generated_text)
    return predicted is not None and predicted == correct_option.strip().upper()
