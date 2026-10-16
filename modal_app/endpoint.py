"""Modal app: deploys a persistent, warm web endpoint per MLLM, plus a
/sweep endpoint that runs the full experimental grid on Modal's own infra —
once spawned, the sweep keeps running (and checkpoints to a Volume) whether
or not your laptop stays connected.

One-time setup
──────────────
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxx   # needs AfriMed-VQA(-Parallel) access

Deploy (stays up until you `modal app stop mcmllm-eval`)
──────────────────────────────────────────────────────────
modal deploy modal_app/endpoint.py

Ad hoc single-item inference (model warms on first call, ~30-90s cold start)
──────────────────────────────────────────────────────────────────────────
curl -X POST https://<workspace>--mcmllm-eval-medgemma4b-predict.modal.run \
     -H 'Content-Type: application/json' \
     -d '{"lang": "swahili", "sample_id": 1, "condition": "multimodal"}'

Kick off the full grid (model × language × condition × seed) — returns
immediately, keeps running on Modal regardless of your machine
────────────────────────────────────────────────────────────────────────
curl -X POST https://<workspace>--mcmllm-eval-sweep-start.modal.run

Check progress
──────────────
curl https://<workspace>--mcmllm-eval-sweep-status.modal.run

Dry run / local convenience (no deploy needed)
────────────────────────────────────────────────
modal run modal_app/endpoint.py --action status
modal run modal_app/endpoint.py --action sweep --langs swahili,hausa --conditions blind

Results
───────
Each (model, lang, condition, seed) trial writes one JSONL file (one line
per item) to the `mcmllm-results` Modal volume. Rerunning the sweep after a
disconnect is safe — already-completed trials are skipped. Then:
  python scripts/aggregate.py
pulls every trial file down, writes results/raw_results.csv, and prints the
Δvisual summary (high- vs. low-resource languages).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

MCMLLM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MCMLLM_ROOT))

from mcmllm.data import CONDITIONS, LANGUAGES, build_eval_inputs, load_language
from mcmllm.models import MODELS
from mcmllm.prompts import build_prompt
from mcmllm.scoring import is_correct, parse_answer

app = modal.App("mcmllm-eval")

# Only the shuffled (mismatched-image) condition has any randomness — which
# "other" image gets attached — so it's the only one worth repeating across
# seeds. Multimodal and blind are deterministic given the question set.
SEEDS = [42, 123, 456]

RESULTS_PATH = "/results"

# `add_local_python_source` must be the *last* step on each image — Modal
# mounts local source at container startup rather than baking it into the
# image layer, so any `pip_install`/`run_commands` after it would force a
# full rebuild on every local file change (and Modal refuses it outright).
_deps = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets>=2.18", "pillow>=10.0", "fastapi[standard]>=0.115",
)

# Lightweight image for the CPU-only orchestration functions below (they
# only need `mcmllm.data` for the language/condition grid, not a GPU stack).
base_image = _deps.add_local_python_source("mcmllm")

# Full stack for the GPU-backed model-serving classes.
gpu_image = (
    _deps.pip_install(
        "torch>=2.6",
        "transformers>=4.52",
        "accelerate>=0.33",
        "timm>=1.0",     # InternVL vision tower
        "einops>=0.8",
    )
    .add_local_python_source("mcmllm")
)

hf_secret   = modal.Secret.from_name("huggingface-secret")
hf_cache    = modal.Volume.from_name("mcmllm-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("mcmllm-results", create_if_missing=True)


def _trial_filename(seed: int) -> str:
    return f"seed_{seed}.jsonl"


def _trial_dir(model_key: str, lang: str, condition: str) -> str:
    return f"{model_key}/{lang}/{condition}"


# ── Shared serving logic, mixed into one `@app.cls` per model below so each
#    model gets its own GPU type / autoscaling pool / web endpoint while
#    sharing identical loading & generation code. ──────────────────────────
class _VLMServerBase:
    model_key: str = ""

    @modal.enter()
    def load(self):
        import torch
        import transformers

        # Newer model cards (e.g. gemma-4-E4B-it) advertise `AutoModelForMultimodalLM`
        # as their auto-class; older-but-current ones (medgemma-4b-it,
        # InternVL3-8B-hf) still load fine through the longer-standing
        # `AutoModelForImageTextToText` alias (confirmed live against both).
        # Try whichever exist in the installed transformers, newest first.
        auto_classes = [
            getattr(transformers, name)
            for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText")
            if hasattr(transformers, name)
        ]

        spec = MODELS[self.model_key]
        attempts = [(spec.hf_repo, spec.trust_remote_code)]
        if spec.fallback_hf_repo:
            attempts.append((spec.fallback_hf_repo, spec.fallback_trust_remote_code))

        last_err: Exception | None = None
        for repo, trust in attempts:
            for auto_cls in auto_classes:
                try:
                    self.processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=trust)
                    self.model = auto_cls.from_pretrained(
                        repo, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=trust,
                    )
                    self.model.eval()
                    return
                except Exception as e:  # noqa: BLE001 — genuinely trying multiple repo/class combos
                    last_err = e

        raise RuntimeError(f"Could not load any repo/auto-class combo for '{self.model_key}'") from last_err

    def _generate(self, image, prompt_text: str) -> str:
        import torch

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        return self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _score(self, ei) -> dict:
        raw_text = self._generate(ei.image, build_prompt(ei.question, ei.options))
        return {
            "model": self.model_key,
            "sample_id": ei.sample_id,
            "lang": ei.lang,
            "condition": ei.condition,
            "correct_option": ei.correct_option,
            "raw_text": raw_text,
            "predicted_option": parse_answer(raw_text),
            "correct": is_correct(raw_text, ei.correct_option),
        }

    @modal.method()
    def predict_one(self, lang: str, sample_id: int, condition: str = "multimodal", seed: int = 42) -> dict:
        items = load_language(lang)
        inputs = build_eval_inputs(items, condition, seed=seed)
        match = next((e for e in inputs if e.sample_id == sample_id), None)
        if match is None:
            raise ValueError(f"sample_id {sample_id} not found in language '{lang}'")
        return self._score(match)

    @modal.fastapi_endpoint(method="POST")
    def predict(self, body: dict):
        return self.predict_one.local(
            lang=body["lang"],
            sample_id=int(body["sample_id"]),
            condition=body.get("condition", "multimodal"),
            seed=int(body.get("seed", 42)),
        )

    @modal.method()
    def run_trial(self, lang: str, condition: str, seed: int) -> dict:
        """Score every item of one (lang, condition, seed) trial and
        checkpoint it to the results volume. Safe to re-run — the caller
        (run_sweep) only invokes this for trials not already on disk.
        """
        items = load_language(lang)
        inputs = build_eval_inputs(items, condition, seed=seed)
        records = [self._score(ei) for ei in inputs]

        out_dir = Path(RESULTS_PATH) / _trial_dir(self.model_key, lang, condition)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _trial_filename(seed)).write_text(
            "\n".join(json.dumps(r) for r in records)
        )
        results_vol.commit()

        accuracy = sum(r["correct"] for r in records) / len(records)
        return {
            "model": self.model_key, "lang": lang, "condition": condition,
            "seed": seed, "n": len(records), "accuracy": round(accuracy, 4),
        }


_CLASS_KWARGS = dict(
    image=gpu_image,
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": hf_cache, RESULTS_PATH: results_vol},
    scaledown_window=300,
    timeout=3600,
    max_containers=6,   # raise if your Modal plan's GPU quota allows more parallelism
)


@app.cls(gpu=MODELS["gemma4-e4b"].gpu, **_CLASS_KWARGS)
class Gemma4E4B(_VLMServerBase):
    model_key = "gemma4-e4b"


@app.cls(gpu=MODELS["medgemma-4b"].gpu, **_CLASS_KWARGS)
class MedGemma4B(_VLMServerBase):
    model_key = "medgemma-4b"


@app.cls(gpu=MODELS["internvl3-8b"].gpu, **_CLASS_KWARGS)
class InternVL38B(_VLMServerBase):
    model_key = "internvl3-8b"


SERVER_CLASSES = {
    "gemma4-e4b": Gemma4E4B,
    "medgemma-4b": MedGemma4B,
    "internvl3-8b": InternVL38B,
}


# ── Sweep orchestration (CPU-only; fans out GPU work via .spawn) ──────────
def _build_jobs(models: list[str], langs: list[str], conditions: list[str]) -> list[tuple]:
    jobs = []
    for model_key in models:
        for lang in langs:
            for condition in conditions:
                trial_seeds = SEEDS if condition == "shuffled" else [SEEDS[0]]
                for seed in trial_seeds:
                    jobs.append((model_key, lang, condition, seed))
    return jobs


def _existing_trial_paths() -> set[str]:
    """One recursive listing of the whole volume, rather than one `listdir`
    per (model, lang, condition) — at ~126 combos, the per-key version blew
    through Modal's VolumeListFiles rate limit and silently (wrongly)
    treated the resulting `ResourceExhaustedError` the same as "no results
    yet", which would have made `run_sweep` re-run (and re-pay for) trials
    that had already completed.
    """
    try:
        entries = results_vol.listdir("/", recursive=True)
    except modal.exception.NotFoundError:
        return set()
    return {e.path for e in entries if e.path.endswith(".jsonl")}


def _completed_jobs(jobs: list[tuple]) -> set[tuple]:
    existing = _existing_trial_paths()
    return {
        job for job in jobs
        if f"{_trial_dir(job[0], job[1], job[2])}/{_trial_filename(job[3])}" in existing
    }


def _status(models=None, langs=None, conditions=None) -> dict:
    models = models or list(MODELS)
    langs = langs or list(LANGUAGES)
    conditions = conditions or list(CONDITIONS)
    jobs = _build_jobs(models, langs, conditions)
    done = _completed_jobs(jobs)
    return {"total": len(jobs), "done": len(done), "pending": len(jobs) - len(done)}


@app.function(image=base_image, volumes={RESULTS_PATH: results_vol}, timeout=24 * 3600)
def run_sweep(models: list[str] | None = None, langs: list[str] | None = None, conditions: list[str] | None = None):
    models = models or list(MODELS)
    langs = langs or list(LANGUAGES)
    conditions = conditions or list(CONDITIONS)

    jobs = _build_jobs(models, langs, conditions)
    done = _completed_jobs(jobs)
    pending = [j for j in jobs if j not in done]
    print(f"Sweep: {len(jobs)} total, {len(done)} already done, {len(pending)} to run.")

    # One model at a time (keeps GPU spend/concurrency bounded); within a
    # model, every (lang, condition, seed) trial is spawned in parallel and
    # auto-scaled across up to max_containers GPU replicas.
    by_model: dict[str, list[tuple]] = {}
    for job in pending:
        by_model.setdefault(job[0], []).append(job)

    for model_key, model_jobs in by_model.items():
        server = SERVER_CLASSES[model_key]()
        calls = [server.run_trial.spawn(lang, condition, seed) for _, lang, condition, seed in model_jobs]
        for call in calls:
            print(call.get())

    print("Sweep complete.")


@app.function(image=base_image, volumes={RESULTS_PATH: results_vol})
@modal.fastapi_endpoint(method="POST")
def sweep_start(body: dict | None = None):
    body = body or {}
    call = run_sweep.spawn(
        models=body.get("models"), langs=body.get("langs"), conditions=body.get("conditions"),
    )
    return {"call_id": call.object_id, "message": "Sweep spawned — safe to close your laptop now."}


@app.function(image=base_image, volumes={RESULTS_PATH: results_vol})
@modal.fastapi_endpoint(method="GET")
def sweep_status():
    return _status()


# ── Local convenience entrypoint (no deploy needed) ────────────────────────
@app.local_entrypoint()
def main(action: str = "status", models: str = "", langs: str = "", conditions: str = ""):
    kwargs = dict(
        models=models.split(",") if models else None,
        langs=langs.split(",") if langs else None,
        conditions=conditions.split(",") if conditions else None,
    )
    if action == "status":
        print(_status(**kwargs))
    elif action == "sweep":
        run_sweep.remote(**kwargs)
    else:
        raise SystemExit(f"Unknown action {action!r}. Use 'status' or 'sweep'.")
