"""
LongMemEvalS Integration for M-flow
====================================
Provides dataset loading, ingestion, retrieval, and scoring utilities
for the LongMemEval-S benchmark (ICLR 2025).

Reference:
    Di Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
    Interactive Memory", ICLR 2025. https://arxiv.org/abs/2410.10813

Usage::

    from m_flow.eval.longmemeval import LongMemEvalRunner
    runner = LongMemEvalRunner(data_path="longmemeval_s.json")
    results = await runner.run(num_questions=50)
    runner.print_report(results)
"""

from .runner import LongMemEvalRunner
from .loader import LongMemEvalLoader, LongMemEvalInstance
from .scorer import LongMemEvalScorer, ScoringResult

__all__ = [
    "LongMemEvalRunner",
    "LongMemEvalLoader",
    "LongMemEvalInstance",
    "LongMemEvalScorer",
    "ScoringResult",
]
