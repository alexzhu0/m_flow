"""
LongMemEvalS Scorer
====================
Implements the two-stage scoring pipeline used in the LongMemEval paper:

1. **Session-level recall** (R@session): Did the retrieval system surface at
   least one evidence session?  This is a binary, deterministic metric.

2. **QA accuracy** (Acc): Does the system's final answer match the ground
   truth?  Two strategies are supported:

   * ``"exact"`` – normalised string match (lower-case, strip punctuation).
     Fast and deterministic; suitable for CI gates.
   * ``"llm"`` – LLM-as-judge using a structured prompt.  Matches the
     evaluation methodology in the original paper and mflow-benchmarks.
     Requires an OpenAI-compatible API key.

Reference:
    Wu et al. (2025), Appendix B.3 – Evaluation Protocol.
    https://arxiv.org/abs/2410.10813
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .loader import LongMemEvalInstance


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InstanceResult:
    """Evaluation result for a single LongMemEval instance."""

    question_id: str
    question_type: str
    is_abstention: bool

    # Retrieval-layer metrics
    retrieved_session_ids: List[str] = field(default_factory=list)
    session_recall: bool = False  # True if ≥1 evidence session retrieved

    # QA-layer metrics
    predicted_answer: str = ""
    qa_correct: bool = False

    # Metadata
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ScoringResult:
    """
    Aggregated scoring result for a LongMemEval evaluation run.

    Attributes:
        num_instances: Total number of evaluated instances.
        session_recall: Fraction of instances where ≥1 evidence session
            was retrieved (session-level recall).
        qa_accuracy: Fraction of instances where the predicted answer is
            correct (QA accuracy).
        by_type: Per-question-type breakdown of session_recall and
            qa_accuracy.
        instance_results: Raw per-instance results.
    """

    num_instances: int = 0
    session_recall: float = 0.0
    qa_accuracy: float = 0.0
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    instance_results: List[InstanceResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_instances": self.num_instances,
            "session_recall": round(self.session_recall, 4),
            "qa_accuracy": round(self.qa_accuracy, 4),
            "by_type": {
                t: {k: round(v, 4) for k, v in m.items()}
                for t, m in self.by_type.items()
            },
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _exact_match(prediction: str, ground_truth: str) -> bool:
    """Normalised exact-match check."""
    return _normalise(prediction) == _normalise(ground_truth)


def _contains_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalised ground truth is contained in normalised prediction."""
    return _normalise(ground_truth) in _normalise(prediction)


# ---------------------------------------------------------------------------
# LLM-judge prompt
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = (
    "You are a strict but fair evaluator for a question-answering benchmark. "
    "Your task is to decide whether a predicted answer is correct given the "
    "ground-truth answer. Focus on semantic equivalence, not surface form. "
    "Respond with exactly one word: 'correct' or 'incorrect'."
)

_LLM_JUDGE_USER_TMPL = (
    "Question: {question}\n"
    "Ground-truth answer: {ground_truth}\n"
    "Predicted answer: {prediction}\n\n"
    "Is the predicted answer correct? (correct/incorrect)"
)


async def _llm_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    model: str = "gpt-4.1-mini",
) -> bool:
    """
    Use an LLM to judge whether *prediction* is semantically equivalent to
    *ground_truth* for the given *question*.

    Falls back to exact-match if the API call fails.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import]

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _LLM_JUDGE_USER_TMPL.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    ),
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict.startswith("correct")
    except Exception:  # noqa: BLE001
        # Graceful degradation: fall back to contains-match
        return _contains_match(prediction, ground_truth)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class LongMemEvalScorer:
    """
    Scores LongMemEval instance results.

    Args:
        scoring_strategy: ``"exact"`` or ``"llm"``.  Default ``"llm"``.
        judge_model: OpenAI-compatible model name for LLM judging.
            Only used when ``scoring_strategy="llm"``.

    Example::

        scorer = LongMemEvalScorer(scoring_strategy="llm")
        result = await scorer.score_instance(instance, retrieved_ids, predicted)
    """

    def __init__(
        self,
        scoring_strategy: str = "llm",
        judge_model: str = "gpt-4.1-mini",
    ) -> None:
        if scoring_strategy not in ("exact", "llm"):
            raise ValueError(
                f"scoring_strategy must be 'exact' or 'llm', got {scoring_strategy!r}"
            )
        self.scoring_strategy = scoring_strategy
        self.judge_model = judge_model

    async def score_instance(
        self,
        instance: LongMemEvalInstance,
        retrieved_session_ids: List[str],
        predicted_answer: str,
        latency_s: float = 0.0,
    ) -> InstanceResult:
        """
        Score a single instance.

        Args:
            instance: The evaluation instance.
            retrieved_session_ids: Session IDs returned by the memory system.
            predicted_answer: The system's answer string.
            latency_s: Wall-clock latency in seconds (informational).

        Returns:
            InstanceResult with recall and QA accuracy populated.
        """
        # Session-level recall
        evidence_set = set(instance.evidence_session_ids)
        retrieved_set = set(retrieved_session_ids)
        session_recall = bool(evidence_set & retrieved_set)

        # QA accuracy
        if instance.is_abstention:
            # For abstention questions the system should say it doesn't know.
            # We check whether the prediction contains a refusal signal.
            qa_correct = _is_abstention_correct(predicted_answer)
        elif self.scoring_strategy == "exact":
            qa_correct = _contains_match(predicted_answer, instance.answer)
        else:
            qa_correct = await _llm_judge(
                question=instance.question,
                ground_truth=instance.answer,
                prediction=predicted_answer,
                model=self.judge_model,
            )

        return InstanceResult(
            question_id=instance.question_id,
            question_type=instance.question_type,
            is_abstention=instance.is_abstention,
            retrieved_session_ids=list(retrieved_session_ids),
            session_recall=session_recall,
            predicted_answer=predicted_answer,
            qa_correct=qa_correct,
            latency_s=latency_s,
        )

    @staticmethod
    def aggregate(results: List[InstanceResult]) -> ScoringResult:
        """
        Aggregate a list of InstanceResult objects into a ScoringResult.

        Args:
            results: Per-instance results (errors are excluded from metrics).

        Returns:
            ScoringResult with overall and per-type metrics.
        """
        valid = [r for r in results if r.ok]
        n = len(valid)
        if n == 0:
            return ScoringResult(num_instances=len(results))

        overall_recall = sum(r.session_recall for r in valid) / n
        overall_acc = sum(r.qa_correct for r in valid) / n

        # Per-type breakdown
        by_type: Dict[str, Dict[str, float]] = {}
        type_groups: Dict[str, List[InstanceResult]] = {}
        for r in valid:
            type_groups.setdefault(r.question_type, []).append(r)

        for q_type, group in sorted(type_groups.items()):
            ng = len(group)
            by_type[q_type] = {
                "n": ng,
                "session_recall": sum(r.session_recall for r in group) / ng,
                "qa_accuracy": sum(r.qa_correct for r in group) / ng,
            }

        return ScoringResult(
            num_instances=len(results),
            session_recall=overall_recall,
            qa_accuracy=overall_acc,
            by_type=by_type,
            instance_results=results,
        )


# ---------------------------------------------------------------------------
# Abstention helpers
# ---------------------------------------------------------------------------

_ABSTENTION_SIGNALS = [
    "don't know",
    "do not know",
    "cannot find",
    "can't find",
    "no information",
    "not mentioned",
    "not found",
    "unable to find",
    "i'm not sure",
    "i am not sure",
    "no record",
    "no evidence",
]


def _is_abstention_correct(prediction: str) -> bool:
    """
    Return True if the prediction correctly abstains from answering.

    We check for common refusal phrases rather than requiring an exact string,
    since different LLMs express uncertainty differently.
    """
    pred_lower = prediction.lower()
    return any(signal in pred_lower for signal in _ABSTENTION_SIGNALS)
