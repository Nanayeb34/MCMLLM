# MCMLLM — Medical grounding vs. language priors in multilingual MLLMs

**Research question:** when an MLLM answers a medical VQA question correctly,
is it reading the image, or riding a text prior? And when it fails in a
low-resource language, is that a vision failure or a language failure?

**Design:** for each of 3 models × 14 languages × 3 conditions, run the same
171 parallel MCQ items from [AfriMed-VQA-Parallel][vqa] and compute

```
Δvisual = Accuracy(multimodal) − Accuracy(blind)
```

then compare Δvisual between the 3 high-resource languages (English, French,
Portuguese) and the 11 low-resource languages.

- **Multimodal** — the item's real image + question + options.
- **Blind** — no image at all (image ablation).
- **Shuffled** — a *different* item's image from the same language pool
  (mismatched-case control), seeded so the pairing is reproducible.

Models: `google/gemma-4-E4B-it`, `google/medgemma-4b-it`,
`OpenGVLab/InternVL3-8B-hf` (falls back to the original
`OpenGVLab/InternVL3-8B` + `trust_remote_code` if the `-hf` conversion isn't
resolvable — see `mcmllm/models.py`).

[vqa]: https://huggingface.co/datasets/afrimedqa/afrimed-vqa-parallel


## Architecture

A single Modal app (`modal_app/endpoint.py`), deployed once with
`modal deploy`, exposing:

- `POST /<model>/predict` — one persistent, warm HTTP endpoint per model
  (3 total), for ad hoc single-item inference. Each model gets its own GPU
  class (`@app.cls`) so it scales and bills independently.
- `POST /sweep-start` — spawns the full (model × lang × condition × seed)
  grid as a Modal Function and returns immediately. The grid then runs
  entirely on Modal's infra — closing your laptop does not stop it.
- `GET /sweep-status` — trial-completion counts, read straight off the
  results volume.

Every trial's item-level predictions are checkpointed as one JSONL file on
the `mcmllm-results` volume; rerunning the sweep skips whatever's already
there, so a crash or a Modal preemption just picks back up.

## One-time setup

```bash
uv sync                 # installs this package's base deps (not torch/transformers — those
                         # only run inside the Modal container image)
modal setup              # if you haven't already authenticated this machine
modal secret list        # confirm `huggingface-secret` exists in your active workspace —
                          # it must contain a token with access to the gated AfriMed-VQA-Parallel
                          # dataset (request access on the dataset page first)
```

## Running it

```bash
# 1. GPU-free sanity check — confirms dataset access and condition construction
python scripts/smoke_test.py

# 2. Deploy
modal deploy modal_app/endpoint.py

# 3. Smoke-test one real prediction before committing to the full grid
curl -X POST https://<workspace>--mcmllm-eval-medgemma4b-predict.modal.run \
     -H 'Content-Type: application/json' \
     -d '{"lang": "english", "sample_id": 1, "condition": "multimodal"}'

# 4. Kick off the full grid (safe to close your laptop right after this returns)
curl -X POST https://<workspace>--mcmllm-eval-sweep-start.modal.run

# 5. Check progress any time
curl https://<workspace>--mcmllm-eval-sweep-status.modal.run

# 6. Pull results + compute the Δvisual summary (re-run any time, even mid-sweep)
python scripts/aggregate.py
```

`scripts/aggregate.py` writes `results/raw_results.csv` (item-level) and
`results/delta_visual_summary.csv` (mean Δvisual per model × resource
group).

## Repo layout

```
mcmllm/
  data.py       # AfriMed-VQA-Parallel loading, high/low-resource language groups,
                 # multimodal/blind/shuffled condition construction
  prompts.py     # fixed MCQ prompt template
  scoring.py     # answer-letter parsing + correctness
  models.py      # model registry (HF repo ids, GPU type, fallback repos) — metadata only,
                 # no torch/transformers import so this stays importable on a laptop
modal_app/
  endpoint.py    # the Modal app: GPU serving classes + sweep orchestration + web endpoints
scripts/
  smoke_test.py  # local, GPU-free dataset/condition check — run this first
  aggregate.py   # pulls the results volume into results/*.csv + prints the Δvisual summary
  significance.py # runs the significant test on the results
```
