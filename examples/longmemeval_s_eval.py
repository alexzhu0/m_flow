#!/usr/bin/env python3
"""
LongMemEval-S Evaluation Example
==================================
Demonstrates how to evaluate M-flow on the LongMemEval-S benchmark
(Wu et al., ICLR 2025).

LongMemEval-S tests five core long-term memory abilities:
    1. Information Extraction      – recall specific facts from past sessions
    2. Multi-Session Reasoning     – synthesise information across sessions
    3. Temporal Reasoning          – answer questions about when events occurred
    4. Knowledge Updates           – handle contradictions / updated facts
    5. Abstention                  – correctly refuse unanswerable questions

Setup
-----
1. Install M-flow::

       pip install mflow-ai

2. Download the dataset::

       pip install huggingface_hub
       huggingface-cli download xiaowu0162/longmemeval \\
           --repo-type dataset --local-dir ./longmemeval_data

3. Set your LLM API key::

       export LLM_API_KEY=sk-...   # or add to .env

4. Run this script::

       python examples/longmemeval_s_eval.py \\
           --data longmemeval_data/longmemeval_s.json \\
           --max-instances 20

Reference
---------
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu.
"LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory."
ICLR 2025. https://arxiv.org/abs/2410.10813
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from m_flow.shared.logging_utils import setup_logging, WARNING


async def run_evaluation(
    data_path: str,
    max_instances: int | None,
    scoring: str,
    output: str | None,
) -> None:
    from m_flow.eval.longmemeval import LongMemEvalRunner

    runner = LongMemEvalRunner(
        data_path=data_path,
        scoring_strategy=scoring,
        judge_model="gpt-4.1-mini",
        top_k=10,
        max_instances=max_instances,
        cleanup_after_instance=True,
        verbose=True,
    )

    print("\nStarting LongMemEval-S evaluation...")
    print("This will ingest each instance's chat history into M-flow,")
    print("build the knowledge graph, and evaluate retrieval + QA.\n")

    result = await runner.run(concurrency=1)

    if output:
        LongMemEvalRunner.save_results(result, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate M-flow on LongMemEval-S (ICLR 2025).",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to longmemeval_s.json",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Evaluate only the first N instances (default: all 500).",
    )
    parser.add_argument(
        "--scoring",
        choices=["llm", "exact"],
        default="llm",
        help="QA scoring strategy (default: llm).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to this JSON file.",
    )
    args = parser.parse_args()

    # Suppress verbose M-flow logs during evaluation
    setup_logging(log_level=WARNING)

    asyncio.run(
        run_evaluation(
            data_path=args.data,
            max_instances=args.max_instances,
            scoring=args.scoring,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
