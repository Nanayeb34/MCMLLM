"""Model registry — metadata only.

Kept free of torch/transformers imports so this module (and anything that
imports `mcmllm.data`/`mcmllm.scoring` alongside it) stays importable on a
plain laptop without the heavy GPU-serving dependencies. The actual
model-loading and generation code lives in `modal_app/endpoint.py`, which
only ever runs inside the Modal container image where those deps exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_repo: str
    gpu: str = "A10G"
    # Fallback repo tried if the primary one 404s — e.g. InternVL3 ships both
    # a `trust_remote_code` original checkpoint and a native-transformers
    # `-hf` conversion; we prefer the latter for reproducibility (no custom
    # modeling code pinned to a specific transformers commit) but keep the
    # original as a safety net in case the `-hf` repo name changes upstream.
    fallback_hf_repo: str | None = None
    trust_remote_code: bool = False
    fallback_trust_remote_code: bool = False


MODELS: dict[str, ModelSpec] = {
    "gemma4-e4b": ModelSpec(hf_repo="google/gemma-4-E4B-it"),
    "medgemma-4b": ModelSpec(hf_repo="google/medgemma-4b-it"),
    "internvl3-8b": ModelSpec(
        hf_repo="OpenGVLab/InternVL3-8B-hf",
        gpu="L40S",
        fallback_hf_repo="OpenGVLab/InternVL3-8B",
        fallback_trust_remote_code=True,
    ),
}
