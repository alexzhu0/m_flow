"""
LoCoMo-10 Quick Evaluation Example
===================================
A minimal, self-contained script that demonstrates how to run M-flow's
EpisodicRetriever against a small subset of LoCoMo questions and score
the results with an LLM judge.

This is NOT the full benchmark script — it is designed as a fast sanity
check (< 5 minutes) to verify your M-flow setup before committing to the
full 10-hour ingestion run.

Full benchmark scripts and authoritative results:
  https://github.com/FlowElement-ai/mflow-benchmarks

Prerequisites
-------------
- M-flow running (Docker or local) with LoCoMo data already ingested
- Environment variables set: LLM_API_KEY, MODEL, EMBEDDING_MODEL
- pip install openai tqdm

Usage
-----
    # Run against the first 20 questions of conversation 0
    python examples/locomo_quick_eval.py \\
        --data-path /path/to/locomo10.json \\
        --conv-idx 0 \\
        --max-questions 20 \\
        --top-k 10

    # Run with a specific dataset name (if you ingested with a custom name)
    python examples/locomo_quick_eval.py \\
        --data-path /path/to/locomo10.json \\
        --conv-idx 0 \\
        --dataset-prefix my_locomo

Important: Stop the M-flow API server before running this script.
KuzuDB does not support concurrent access. Running search while the
Gunicorn server is active causes silent retrieval failures (~9% of
questions return 0 memories, reducing scores by ~8 percentage points).

Category Mapping (locomo10.json)
---------------------------------
The category numbers in the dataset do NOT match the paper's text order.
Correct mapping (source: snap-research/locomo Issue #27):
    1 → Multi-hop      (282 questions)
    2 → Temporal       (321 questions)
    3 → Open-domain    (96 questions)
    4 → Single-hop     (841 questions)
    5 → Adversarial    (446 questions, excluded — no gold answers)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Category mapping (authoritative — from snap-research/locomo Issue #27)
# ---------------------------------------------------------------------------
CATEGORY_MAPPING: dict[str, str] = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
    "5": "Adversarial",  # excluded from scoring
}

# ---------------------------------------------------------------------------
# LLM judge prompt (Mem0's published ACCURACY_PROMPT — used in the benchmark)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are an expert evaluator. Given a question, the correct answer, and a \
predicted answer, determine if the predicted answer is correct.

Question: {question}
Correct Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Respond with a JSON object: {{"score": 1}} if correct, {{"score": 0}} if wrong.
A predicted answer is correct if it conveys the same essential information as \
the correct answer, even if worded differently.
"""

ANSWER_PROMPT = """\
You are a helpful assistant answering questions based on the provided memories.

Memories:
{memories}

Question: {question}

Instructions:
- Answer based only on the provided memories.
- Be concise and direct.
- If the memories do not contain enough information, say so.
- Do not stop at the first match — combine evidence from multiple memories.
- Look for direct mentions and indirect implications.
"""


# ---------------------------------------------------------------------------
# M-flow search helper
# ---------------------------------------------------------------------------
async def search_episodic(
    query: str,
    dataset_name: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve episodic memories for a query using EpisodicRetriever.

    Returns a list of dicts with keys: memory (str), timestamp (str), score (float).

    NOTE: We use EpisodicRetriever directly (not m_flow.search()) because the
    high-level API returns a formatted string, not the raw Edge objects needed
    to extract Episode summaries and timestamps.
    """
    from m_flow.retrieval.episodic_retriever import EpisodicRetriever
    from m_flow.retrieval.episodic import EpisodicConfig
    from m_flow.context_global_variables import set_db_context
    from m_flow.data.methods import get_datasets_by_name
    from m_flow.auth.methods.get_seed_user import get_seed_user

    seed_user = await get_seed_user()
    datasets = await get_datasets_by_name(dataset_name, user_id=seed_user.id)
    if not datasets:
        return []

    dataset_ids = [str(d.id) for d in datasets]
    set_db_context(user_id=str(seed_user.id), dataset_ids=dataset_ids)

    config = EpisodicConfig(top_k=top_k, wide_search_top_k=top_k * 3)
    retriever = EpisodicRetriever(config=config)
    edges = await retriever.retrieve(query=query, dataset_ids=dataset_ids)

    memories: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        for node in (getattr(edge, "node1", None), getattr(edge, "node2", None)):
            if node is None:
                continue
            if node.attributes.get("type") != "Episode":
                continue
            node_id = str(node.id)
            if node_id in seen:
                continue
            seen.add(node_id)
            summary = node.attributes.get("summary", "")
            ts_ms = node.attributes.get("mentioned_time_start_ms")
            timestamp = ""
            if ts_ms is not None:
                from datetime import datetime, timezone
                timestamp = datetime.fromtimestamp(
                    ts_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            memories.append(
                {
                    "memory": summary,
                    "timestamp": timestamp,
                    "score": float(getattr(edge, "score", 0.0)),
                }
            )
    return memories


# ---------------------------------------------------------------------------
# Answer + judge helpers (synchronous, using OpenAI)
# ---------------------------------------------------------------------------
def generate_answer(client: Any, question: str, memories: list[dict]) -> str:
    """Generate an answer from retrieved memories."""
    if not memories:
        return "No relevant memories found."

    memory_text = "\n".join(
        f"[{i + 1}] {m['memory']}" + (f" ({m['timestamp']})" if m["timestamp"] else "")
        for i, m in enumerate(memories)
    )
    response = client.chat.completions.create(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    memories=memory_text, question=question
                ),
            }
        ],
    )
    return response.choices[0].message.content.strip()


def judge_answer(
    client: Any, question: str, gold: str, predicted: str, judge_model: str
) -> int:
    """Return 1 if predicted is correct, 0 otherwise."""
    response = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    gold_answer=gold,
                    predicted_answer=predicted,
                ),
            }
        ],
    )
    try:
        return int(json.loads(response.choices[0].message.content).get("score", 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def run_eval(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI()

    # Load dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    conversations = dataset if isinstance(dataset, list) else dataset.get("conversations", [])
    if args.conv_idx >= len(conversations):
        raise ValueError(
            f"conv_idx {args.conv_idx} out of range (dataset has {len(conversations)} conversations)"
        )

    conv = conversations[args.conv_idx]
    qa_pairs = conv.get("qa_pairs", conv.get("questions", []))
    # Exclude category 5 (Adversarial — no gold answers)
    qa_pairs = [q for q in qa_pairs if str(q.get("category", "")) != "5"]

    if args.max_questions:
        qa_pairs = qa_pairs[: args.max_questions]

    speakers = conv.get("speakers", [f"speaker_{args.conv_idx}_0", f"speaker_{args.conv_idx}_1"])
    dataset_name = f"{args.dataset_prefix}_{args.conv_idx}"

    print(f"\nM-flow LoCoMo Quick Eval")
    print(f"  Conversation : {args.conv_idx} ({' ↔ '.join(speakers)})")
    print(f"  Questions    : {len(qa_pairs)}")
    print(f"  Dataset name : {dataset_name}")
    print(f"  Top-K        : {args.top_k}")
    print(f"  Judge model  : {args.judge_model}")
    print()

    results: list[dict] = []
    by_category: dict[str, list[int]] = defaultdict(list)

    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        gold = qa.get("answer", qa.get("gold_answer", ""))
        category = str(qa.get("category", ""))
        category_name = CATEGORY_MAPPING.get(category, f"cat_{category}")

        t0 = time.time()
        memories = await search_episodic(question, dataset_name, top_k=args.top_k)
        search_time = time.time() - t0

        predicted = generate_answer(client, question, memories)
        score = judge_answer(client, question, gold, predicted, args.judge_model)

        by_category[category_name].append(score)
        results.append(
            {
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "category": category,
                "category_name": category_name,
                "num_memories": len(memories),
                "llm_judge_score": score,
                "search_time_s": round(search_time, 2),
            }
        )

        status = "✓" if score == 1 else "✗"
        print(
            f"  [{i + 1:3d}/{len(qa_pairs)}] {status} cat={category_name:<12} "
            f"mem={len(memories):2d}  {question[:60]}"
        )

    # Summary
    total = len(results)
    correct = sum(r["llm_judge_score"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"  Overall: {correct}/{total} = {correct / total:.1%}")
    print()
    for cat, scores in sorted(by_category.items()):
        n = len(scores)
        c = sum(scores)
        print(f"  {cat:<14}: {c}/{n} = {c / n:.1%}")
    print(f"{'=' * 60}\n")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conv_idx": args.conv_idx,
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total if total else 0,
                    "by_category": {
                        k: {"correct": sum(v), "total": len(v), "accuracy": sum(v) / len(v)}
                        for k, v in by_category.items()
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Results saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo-10 quick evaluation against M-flow EpisodicRetriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--conv-idx",
        type=int,
        default=0,
        help="Conversation index to evaluate (0–9, default: 0)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve (default: 10, matches benchmark)",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="locomo_benchmark",
        help="Dataset name prefix used during ingestion (default: locomo_benchmark)",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM model for judging (default: gpt-4o-mini, matches benchmark)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (optional)",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
