"""Prompt construction, kept identical across languages and conditions.

The instruction wrapper is fixed in English for every language so that only
the question/options text varies by language — translating the instruction
too would add another source of variation between the high- and
low-resource conditions and confound the comparison this study is making.
"""

from __future__ import annotations

from mcmllm.data import OPTION_LETTERS

INSTRUCTION = (
    "Answer the following multiple-choice medical question. "
    "Respond with only the letter of the correct option (A, B, C, or D) "
    "and nothing else."
)


def build_prompt(question: str, options: dict[str, str]) -> str:
    option_lines = "\n".join(f"{letter}. {options[letter]}" for letter in OPTION_LETTERS)
    return f"{INSTRUCTION}\n\nQuestion: {question}\n{option_lines}\nAnswer:"
