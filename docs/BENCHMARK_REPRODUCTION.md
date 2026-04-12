# LoCoMo-10 Benchmark — Reproduction Guide

This document explains how to reproduce the M-flow benchmark results reported in the README
(81.8% LLM-Judge accuracy on LoCoMo-10).

Full scripts, per-conversation results, and methodology details live in the companion
repository: [FlowElement-ai/mflow-benchmarks](https://github.com/FlowElement-ai/mflow-benchmarks).

---

## Quick Reference

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo-10 (10 conversations, 1 541 evaluated questions) |
| M-flow version | GitHub `main` ≥ commit `3afcb94` |
| LLM — ingestion | `gpt-5-nano` |
| LLM — answer | `gpt-5-mini` (temperature = 1, model default) |
| LLM — judge | `gpt-4o-mini` (temperature = 0) |
| Embedding | `text-embedding-3-small` (1 536 dim) |
| Retrieval | `EpisodicRetriever`, `top_k = 10` |
| Precise mode | **enabled** (`precise_mode = True`) |
| Graph DB | KuzuDB |
| Vector DB | LanceDB |

---

## Category Mapping

> **Common mistake**: many implementations assume Cat 1 = Single-hop based on the paper's
> text order. The actual mapping in `locomo10.json` is different.

The correct mapping (from [snap-research/locomo Issue #27](https://github.com/snap-research/LoCoMo/issues/27)):

| Category number in dataset | Question type | Count |
|:--------------------------:|---------------|------:|
| 1 | Multi-hop | 282 |
| 2 | Temporal | 321 |
| 3 | Open-domain | 96 |
| 4 | Single-hop | 841 |
| 5 | Adversarial (excluded — no gold answers) | 446 |
| **Total evaluated** | | **1 541** |

Always use `config/category_mapping.json` from the benchmarks repository, not the paper's
text ordering.

---

## Prerequisites

- Docker (for running M-flow as an isolated service)
- Python 3.10+ with `openai`, `tqdm`, `python-dotenv`, `jinja2`
- OpenAI API key (for answer generation and judging)
- `locomo10.json` — see `data/DATA_SOURCE.md` in the benchmarks repository

---

## Step-by-Step Reproduction

### Step 1 — Use the correct M-flow version

M-flow 0.3.2 on PyPI contains a bug where `config` is undefined inside
`_task_generate_facets()`, causing Episode summaries to degrade to raw text truncated at
500 characters. This significantly lowers benchmark scores.

**Always clone from GitHub main (commit `3afcb94` or later):**

```bash
git clone https://github.com/FlowElement-ai/m_flow.git
cd m_flow
# Verify you are on a commit after 3afcb94
git log --oneline -5
```

**Verification during ingestion**: container logs should show lines like:

```
Generated N sections, M facets
```

If you instead see `summarize_by_event failed: name 'config' is not defined`, you are
running an affected version.

### Step 2 — Build the Docker image

```bash
docker build -t m_flow_local .
```

### Step 3 — Ingest LoCoMo-10

Use the batched ingest script from the benchmarks repository with **precise mode enabled**:

```bash
python benchmarks/locomo/scripts_original/run_ingest_batched.py \
  --data-path /path/to/locomo10.json \
  --no-prune \
  --force
```

Key ingestion parameters (passed in the API payload):

```json
{
  "enable_episode_routing": true,
  "precise_mode": true
}
```

> **Why precise mode matters**: without it, summarization uses lossy compression and
> benchmark scores drop by several percentage points.

Ingestion time: approximately **10 hours 39 minutes** for all 10 conversations
(272 sessions, 3-3-4 parallel batching strategy).

### Step 4 — Stop the API server before searching

M-flow uses KuzuDB, an embedded graph database that **does not support concurrent access**.
Running the search script while the Gunicorn API server is active causes ~9% of questions
to return zero memories silently, reducing the overall score by ~8 percentage points.

```bash
# Stop the running API server
docker stop <m_flow_container>

# Run search via docker run (not docker exec) against the committed image
docker run --rm \
  -e LLM_API_KEY="sk-..." \
  -e MODEL="gpt-5-mini" \
  -e EMBEDDING_MODEL="openai/text-embedding-3-small" \
  -e EMBEDDING_DIMENSIONS=1536 \
  --entrypoint /opt/m_flow/.venv/bin/python \
  m_flow_with_data \
  /opt/m_flow/search_aligned.py \
    --data-path /opt/m_flow/locomo10.json \
    --output-path /opt/m_flow/results/mflow_search_conv0.json \
    --top-k 10 \
    --max-conversations 1
```

Repeat for each conversation (change `--max-conversations` and the output filename).

### Step 5 — Handle timeouts

`gpt-5-mini` occasionally times out (~1–5% of questions per conversation). Retry failed
questions using the **identical** prompt from `answer_question()` — do not use a simplified
prompt, as that reduces accuracy for the retried questions.

The reference run had 19 retried questions (1.2% of 1 541) using a simplified prompt.
Future runs should implement retry logic that reuses the exact same prompt template.

### Step 6 — Evaluate

```bash
python benchmarks/locomo/scripts/evaluate_aligned.py \
  --input-file results/mflow_search_conv0.json \
  --output-file results/mflow_eval_conv0.json \
  --model gpt-4o-mini
```

The judge uses Mem0's published `ACCURACY_PROMPT` (generous grading). Results are
non-deterministic for the answer step (gpt-5-mini temperature = 1); expect ±1–2% variance
between runs. Conv 8 variance test: mean 75.1%, std 0.9%, range 73.7%–76.3%.

---

## Known Limitations

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | `gpt-5-mini` temperature = 1 | ±1–2% variance between runs |
| 2 | 19 timeout retries used simplified prompt | Slight accuracy reduction for 1.2% of questions |
| 3 | No `max_tokens` set for answer or judge LLM | Uses model defaults |
| 4 | Category mapping differs from paper text order | Use `config/category_mapping.json` |

---

## Results Summary

| Conv | Speakers | Questions | Score |
|:----:|----------|:---------:|------:|
| 0 | Caroline ↔ Melanie | 152 | 83.6% |
| 1 | Jon ↔ Gina | 82 | 90.2% |
| 2 | John ↔ Maria | 152 | 86.2% |
| 3 | Joanna ↔ Nate | 199 | 78.9% |
| 4 | Tim ↔ John | 178 | 77.5% |
| 5 | Audrey ↔ Andrew | 123 | 79.7% |
| 6 | James ↔ John | 150 | 88.7% |
| 7 | Deborah ↔ Jolene | 191 | 80.6% |
| 8 | Evan ↔ Sam | 156 | 75.6% |
| 9 | Calvin ↔ Dave | 158 | 82.9% |
| **Total** | | **1 541** | **81.8%** |

Per-category breakdown:

| Category | Type | Correct | Total | Score |
|:--------:|------|:-------:|:-----:|------:|
| 4 | Single-hop | 737 | 841 | 87.6% |
| 2 | Temporal | 255 | 321 | 79.4% |
| 1 | Multi-hop | 212 | 282 | 75.2% |
| 3 | Open-domain | 56 | 96 | 58.3% |

---

## Comparison with Other Systems

All systems below use the same judge LLM (`gpt-4o-mini`) and answer LLM (`gpt-5-mini`)
for a fair comparison at `top_k = 10`:

| System | Score | Top-K |
|--------|:-----:|:-----:|
| **M-flow** | **81.8%** | 10 |
| Cognee Cloud | 79.4% | 10 |
| Zep Cloud | 73.4% | 10 |
| Supermemory | 64.4% | 10 |

Full methodology and per-system reproduction scripts are in
[FlowElement-ai/mflow-benchmarks](https://github.com/FlowElement-ai/mflow-benchmarks).

---

## Frequently Asked Questions

**Q: I'm getting lower scores than 81.8%. What should I check?**

1. Confirm you are using GitHub `main` at commit `3afcb94` or later (not PyPI 0.3.2).
2. Confirm `precise_mode = True` during ingestion.
3. Confirm the M-flow API server is **stopped** before running the search script.
4. Confirm you are using the correct category mapping (Cat 4 = Single-hop, not Cat 1).

**Q: The search script returns zero memories for some questions.**

This is almost always the KuzuDB file lock issue. Stop the API server before running the
search script (see Step 4).

**Q: Can I use a different LLM for answer generation?**

Yes, but results will not be directly comparable to the reported 81.8%. The benchmark uses
`gpt-5-mini` at temperature = 1 (non-configurable). Using a different model or temperature
will produce different scores.

**Q: Why is Category 3 (Open-domain) accuracy lower (58.3%)?**

Open-domain questions require synthesizing information across multiple conversations or
drawing on general knowledge not present in the memory store. This is a known limitation
of retrieval-augmented approaches and is consistent with results from other systems.

---

## See Also

- [Retrieval Architecture](RETRIEVAL_ARCHITECTURE.md) — how Bundle Search works
- [mflow-benchmarks](https://github.com/FlowElement-ai/mflow-benchmarks) — full scripts,
  raw results, and per-system comparisons
- [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo) — original dataset
