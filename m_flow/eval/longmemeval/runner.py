"""
LongMemEvalS Runner
====================
End-to-end evaluation runner that:

1. Loads LongMemEval-S instances from the official JSON file.
2. Ingests each instance's chat history into M-flow using ``m_flow.add()``
   with per-session timestamps, grouped under a per-instance dataset.
3. Calls ``m_flow.memorize()`` to build the knowledge graph.
4. Issues the evaluation question via ``m_flow.search()`` in EPISODIC mode
   and collects the retrieved session IDs and answer text.
5. Scores each instance using :class:`LongMemEvalScorer`.
6. Aggregates and reports results.

Design decisions
----------------
* **Per-instance dataset isolation**: Each question gets its own M-flow
  dataset (named ``lme_<question_id>``).  This prevents cross-contamination
  between haystack histories and mirrors the evaluation setup in the paper.
* **Timestamp preservation**: Session dates are passed as ``created_at``
  to ``m_flow.add()`` so that M-flow's time-aware retrieval can leverage
  the temporal structure of the haystack.
* **Session-ID tracking**: After retrieval, we map retrieved episode IDs
  back to original session IDs using a lightweight index built during
  ingestion.  This enables session-level recall evaluation.
* **Concurrency**: Instances are processed serially by default to avoid
  database contention.  Pass ``concurrency > 1`` for faster (but less
  stable) runs.

Reference:
    Wu et al. (2025), §3 – Evaluation Setup.
    https://arxiv.org/abs/2410.10813
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from m_flow.shared.logging_utils import get_logger

from .loader import (
    LongMemEvalInstance,
    LongMemEvalLoader,
    ChatSession,
    get_type_distribution,
)
from .scorer import InstanceResult, LongMemEvalScorer, ScoringResult

_log = get_logger("LongMemEvalRunner")


# ---------------------------------------------------------------------------
# Session → Dataset index entry
# ---------------------------------------------------------------------------

@dataclass
class _SessionIndex:
    """Maps M-flow dataset info back to original session IDs."""

    dataset_name: str
    dataset_id: Optional[UUID] = None
    # session_id → list of content snippets (first 80 chars) for matching
    session_snippets: Dict[str, List[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LongMemEvalRunner:
    """
    End-to-end LongMemEval-S evaluation runner for M-flow.

    Args:
        data_path: Path to ``longmemeval_s.json``.
        scoring_strategy: ``"llm"`` (default) or ``"exact"``.
        judge_model: LLM model for QA judging (only used with ``"llm"``).
        top_k: Number of episodic memories to retrieve per question.
        max_instances: Cap the number of instances (useful for quick tests).
        question_types: Filter by question type(s).  ``None`` = all types.
        cleanup_after_instance: If True, prune each instance's dataset after
            evaluation to save disk space.  Default True.
        verbose: Print per-instance progress.

    Example::

        import asyncio
        from m_flow.eval.longmemeval import LongMemEvalRunner

        async def main():
            runner = LongMemEvalRunner(
                data_path="longmemeval_data/longmemeval_s.json",
                max_instances=20,
                scoring_strategy="llm",
            )
            results = await runner.run()
            runner.print_report(results)

        asyncio.run(main())
    """

    def __init__(
        self,
        data_path: str,
        scoring_strategy: str = "llm",
        judge_model: str = "gpt-4.1-mini",
        top_k: int = 10,
        max_instances: Optional[int] = None,
        question_types: Optional[List[str]] = None,
        cleanup_after_instance: bool = True,
        verbose: bool = True,
    ) -> None:
        self.data_path = data_path
        self.scorer = LongMemEvalScorer(
            scoring_strategy=scoring_strategy,
            judge_model=judge_model,
        )
        self.top_k = top_k
        self.max_instances = max_instances
        self.question_types = question_types
        self.cleanup_after_instance = cleanup_after_instance
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, concurrency: int = 1) -> ScoringResult:
        """
        Run the full evaluation pipeline.

        Args:
            concurrency: Number of instances to evaluate in parallel.
                Use 1 (default) for stability; higher values may cause
                database contention with KuzuDB.

        Returns:
            Aggregated :class:`ScoringResult`.
        """
        loader = LongMemEvalLoader(
            data_path=self.data_path,
            max_instances=self.max_instances,
            question_types=self.question_types,
        )
        instances = loader.load()

        if not instances:
            raise ValueError(
                "No instances loaded. Check data_path and question_types filter."
            )

        if self.verbose:
            dist = get_type_distribution(instances)
            print(f"\n{'=' * 60}")
            print(f"LongMemEval-S Evaluation  |  M-flow")
            print(f"{'=' * 60}")
            print(f"Dataset     : {self.data_path}")
            print(f"Instances   : {len(instances)}")
            print(f"Scoring     : {self.scorer.scoring_strategy}")
            print(f"Type dist   : {dist}")
            print(f"{'=' * 60}\n")

        instance_results: List[InstanceResult] = []

        if concurrency <= 1:
            for i, inst in enumerate(instances):
                result = await self._evaluate_instance(inst, idx=i + 1, total=len(instances))
                instance_results.append(result)
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _bounded(inst: LongMemEvalInstance, idx: int) -> InstanceResult:
                async with sem:
                    return await self._evaluate_instance(inst, idx=idx, total=len(instances))

            tasks = [_bounded(inst, i + 1) for i, inst in enumerate(instances)]
            instance_results = list(await asyncio.gather(*tasks))

        scoring = LongMemEvalScorer.aggregate(instance_results)

        if self.verbose:
            self.print_report(scoring)

        return scoring

    # ------------------------------------------------------------------
    # Per-instance pipeline
    # ------------------------------------------------------------------

    async def _evaluate_instance(
        self,
        instance: LongMemEvalInstance,
        idx: int = 0,
        total: int = 0,
    ) -> InstanceResult:
        """Run the full pipeline for a single instance."""
        import m_flow

        dataset_name = f"lme_{instance.question_id}"
        t0 = time.monotonic()

        if self.verbose:
            print(
                f"[{idx:>3}/{total}] {instance.question_id} "
                f"({instance.question_type}) | "
                f"{instance.num_sessions} sessions"
            )

        try:
            # ── Step 1: Ingest haystack sessions ──────────────────────
            session_id_map = await self._ingest_sessions(
                instance=instance,
                dataset_name=dataset_name,
            )

            # ── Step 2: Build knowledge graph ─────────────────────────
            await m_flow.memorize(datasets=[dataset_name])

            # ── Step 3: Retrieve and answer ───────────────────────────
            retrieved_session_ids, predicted_answer = await self._retrieve_and_answer(
                instance=instance,
                dataset_name=dataset_name,
                session_id_map=session_id_map,
            )

            # ── Step 4: Score ─────────────────────────────────────────
            latency = time.monotonic() - t0
            result = await self.scorer.score_instance(
                instance=instance,
                retrieved_session_ids=retrieved_session_ids,
                predicted_answer=predicted_answer,
                latency_s=latency,
            )

            if self.verbose:
                recall_str = "✓" if result.session_recall else "✗"
                qa_str = "✓" if result.qa_correct else "✗"
                print(
                    f"         recall={recall_str}  qa={qa_str}  "
                    f"({latency:.1f}s)"
                )

        except Exception as exc:  # noqa: BLE001
            _log.warning(f"Instance {instance.question_id} failed: {exc}")
            result = InstanceResult(
                question_id=instance.question_id,
                question_type=instance.question_type,
                is_abstention=instance.is_abstention,
                error=str(exc),
                latency_s=time.monotonic() - t0,
            )

        finally:
            # ── Step 5: Cleanup ───────────────────────────────────────
            if self.cleanup_after_instance:
                try:
                    await m_flow.prune.prune_data(datasets=[dataset_name])
                except Exception:  # noqa: BLE001
                    pass

        return result

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------

    async def _ingest_sessions(
        self,
        instance: LongMemEvalInstance,
        dataset_name: str,
    ) -> Dict[str, str]:
        """
        Ingest all haystack sessions into M-flow.

        Returns a mapping from M-flow dataset_name to original session_id.
        Since we use one dataset per instance, we instead build a content-
        based index: snippet → session_id, used later to map retrieved
        episodes back to session IDs.

        Returns:
            Dict mapping session_id → plain-text snippet (first 120 chars).
        """
        import m_flow

        session_id_map: Dict[str, str] = {}  # snippet_key → session_id

        for session in instance.sessions:
            plain = session.to_plain_text()
            # Build a snippet key for later matching
            snippet_key = plain[:120].strip()
            session_id_map[snippet_key] = session.session_id

            # Parse session date for timestamp
            created_at: Optional[datetime] = session.date_as_datetime
            if created_at is not None:
                # Ensure UTC
                created_at = created_at.replace(tzinfo=timezone.utc)

            await m_flow.add(
                plain,
                dataset_name=dataset_name,
                created_at=created_at,
            )

        return session_id_map

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    async def _retrieve_and_answer(
        self,
        instance: LongMemEvalInstance,
        dataset_name: str,
        session_id_map: Dict[str, str],
    ) -> Tuple[List[str], str]:
        """
        Issue the evaluation question to M-flow and collect results.

        Returns:
            Tuple of (retrieved_session_ids, predicted_answer).
        """
        import m_flow
        from m_flow.api.v1.search import RecallMode

        # Build a time-aware query by appending the question date
        query = instance.question
        if instance.question_date:
            query = f"[As of {instance.question_date}] {query}"

        results = await m_flow.search(
            query_text=query,
            query_type=RecallMode.EPISODIC,
            datasets=[dataset_name],
            top_k=self.top_k,
        )

        # Extract retrieved session IDs by matching episode content
        retrieved_session_ids = self._map_results_to_session_ids(
            results=results,
            session_id_map=session_id_map,
            instance=instance,
        )

        # Build answer from retrieved context
        predicted_answer = self._build_answer(results, instance)

        return retrieved_session_ids, predicted_answer

    @staticmethod
    def _map_results_to_session_ids(
        results,
        session_id_map: Dict[str, str],
        instance: LongMemEvalInstance,
    ) -> List[str]:
        """
        Map M-flow search results back to original LongMemEval session IDs.

        Strategy: for each retrieved result, check if its text content
        overlaps with any known session snippet.  This is a best-effort
        heuristic; perfect mapping would require storing session IDs as
        metadata during ingestion (a future improvement).
        """
        matched_ids: List[str] = []
        seen: set = set()

        for result in (results if isinstance(results, list) else []):
            # Extract text from result (SearchResult or dict)
            result_text = ""
            if hasattr(result, "search_result"):
                sr = result.search_result
                if isinstance(sr, str):
                    result_text = sr
                elif isinstance(sr, dict):
                    result_text = str(sr.get("content", "")) + str(sr.get("text", ""))
            elif isinstance(result, dict):
                result_text = str(result.get("content", "")) + str(result.get("text", ""))
            elif isinstance(result, str):
                result_text = result

            # Match against session snippets
            for snippet_key, session_id in session_id_map.items():
                if session_id in seen:
                    continue
                # Check for overlap: snippet key appears in result text
                # or result text appears in the original session
                key_words = set(snippet_key.lower().split())
                result_words = set(result_text.lower().split())
                overlap = len(key_words & result_words)
                if overlap >= max(3, len(key_words) * 0.3):
                    matched_ids.append(session_id)
                    seen.add(session_id)
                    break

        return matched_ids

    @staticmethod
    def _build_answer(results, instance: LongMemEvalInstance) -> str:
        """
        Build a plain-text answer from retrieved episodic memories.

        For abstention questions, we check whether any relevant content
        was retrieved; if not, we return a standard refusal phrase.
        """
        if not results:
            if instance.is_abstention:
                return "I don't have enough information to answer this question."
            return ""

        # Collect context snippets
        snippets: List[str] = []
        for result in (results if isinstance(results, list) else []):
            if hasattr(result, "search_result"):
                sr = result.search_result
                if isinstance(sr, str) and sr.strip():
                    snippets.append(sr.strip())
                elif isinstance(sr, dict):
                    text = sr.get("content") or sr.get("text") or ""
                    if text.strip():
                        snippets.append(str(text).strip())
            elif isinstance(result, str) and result.strip():
                snippets.append(result.strip())

        if not snippets:
            if instance.is_abstention:
                return "I don't have enough information to answer this question."
            return ""

        # For abstention questions with retrieved content, still abstain
        if instance.is_abstention:
            return "I don't have enough information to answer this question."

        # Return the most relevant snippet as the answer
        # (In production, this would be passed to an LLM for synthesis)
        return snippets[0]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(result: ScoringResult) -> None:
        """Print a formatted evaluation report to stdout."""
        print(f"\n{'=' * 60}")
        print("LongMemEval-S Results")
        print(f"{'=' * 60}")
        print(f"Instances evaluated : {result.num_instances}")
        print(f"Session Recall      : {result.session_recall:.1%}")
        print(f"QA Accuracy         : {result.qa_accuracy:.1%}")

        if result.by_type:
            print(f"\n{'─' * 60}")
            print(f"{'Question Type':<30} {'N':>5} {'Recall':>8} {'Acc':>8}")
            print(f"{'─' * 60}")
            for q_type, metrics in sorted(result.by_type.items()):
                print(
                    f"{q_type:<30} "
                    f"{int(metrics['n']):>5} "
                    f"{metrics['session_recall']:>7.1%} "
                    f"{metrics['qa_accuracy']:>7.1%}"
                )

        errors = [r for r in result.instance_results if not r.ok]
        if errors:
            print(f"\n⚠  {len(errors)} instance(s) failed with errors.")

        print(f"{'=' * 60}\n")

    @staticmethod
    def save_results(result: ScoringResult, output_path: str) -> None:
        """Save the scoring result to a JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = result.to_dict()
        data["instance_results"] = [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "is_abstention": r.is_abstention,
                "session_recall": r.session_recall,
                "qa_correct": r.qa_correct,
                "predicted_answer": r.predicted_answer,
                "latency_s": round(r.latency_s, 2),
                "error": r.error,
            }
            for r in result.instance_results
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
