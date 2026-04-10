"""
LongMemEval-S CLI entry point.

Usage::

    python -m m_flow.eval.longmemeval \\
        --data longmemeval_data/longmemeval_s.json \\
        --max-instances 20 \\
        --scoring llm \\
        --output results/lme_s_results.json

    # Quick smoke test (exact match, 5 instances)
    python -m m_flow.eval.longmemeval \\
        --data longmemeval_data/longmemeval_s.json \\
        --max-instances 5 \\
        --scoring exact

    # Evaluate only temporal-reasoning questions
    python -m m_flow.eval.longmemeval \\
        --data longmemeval_data/longmemeval_s.json \\
        --types temporal-reasoning multi-session \\
        --scoring llm

Download the dataset first::

    pip install huggingface_hub
    huggingface-cli download xiaowu0162/longmemeval \\
        --repo-type dataset --local-dir ./longmemeval_data
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .loader import LongMemEvalLoader, QUESTION_TYPES
from .runner import LongMemEvalRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m m_flow.eval.longmemeval",
        description="Evaluate M-flow on the LongMemEval-S benchmark (ICLR 2025).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        metavar="PATH",
        help="Path to longmemeval_s.json (or longmemeval_m.json).",
    )
    parser.add_argument(
        "--max-instances",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N instances (useful for quick tests).",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=sorted(QUESTION_TYPES),
        default=None,
        metavar="TYPE",
        help=(
            "Filter by question type(s). "
            f"Choices: {', '.join(sorted(QUESTION_TYPES))}. "
            "Default: all types."
        ),
    )
    parser.add_argument(
        "--scoring",
        choices=["llm", "exact"],
        default="llm",
        help=(
            "QA scoring strategy. "
            "'llm' uses an LLM judge (matches paper methodology). "
            "'exact' uses normalised string containment (fast, deterministic). "
            "Default: llm."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        metavar="MODEL",
        help="LLM model for judging (only used with --scoring llm). Default: gpt-4.1-mini.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Number of episodic memories to retrieve per question. Default: 10.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Save results to this JSON file path.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of instances to evaluate in parallel. "
            "Use 1 (default) for stability with KuzuDB."
        ),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not prune per-instance datasets after evaluation.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-instance progress output.",
    )
    parser.add_argument(
        "--download-instructions",
        action="store_true",
        help="Print dataset download instructions and exit.",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if args.download_instructions:
        print(LongMemEvalLoader.get_download_instructions())
        return 0

    runner = LongMemEvalRunner(
        data_path=args.data,
        scoring_strategy=args.scoring,
        judge_model=args.judge_model,
        top_k=args.top_k,
        max_instances=args.max_instances,
        question_types=args.types,
        cleanup_after_instance=not args.no_cleanup,
        verbose=not args.quiet,
    )

    result = await runner.run(concurrency=args.concurrency)

    if args.output:
        LongMemEvalRunner.save_results(result, args.output)

    # Exit code: 0 if QA accuracy ≥ 50%, 1 otherwise (useful for CI)
    return 0 if result.qa_accuracy >= 0.5 else 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
