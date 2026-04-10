"""
LongMemEvalS Dataset Loader
============================
Loads and validates LongMemEval-S instances from the official JSON format.

Dataset format (from xiaowu0162/LongMemEval):
    - question_id: str
    - question_type: one of {single-session-user, single-session-assistant,
        single-session-preference, temporal-reasoning, knowledge-update,
        multi-session}. Abstention questions end with "_abs".
    - question: str
    - answer: str
    - question_date: str  (ISO date, e.g. "2024-03-15")
    - haystack_session_ids: List[str]
    - haystack_dates: List[str]
    - haystack_sessions: List[List[Dict]]  (each session is a list of turns)
    - answer_session_ids: List[str]  (evidence sessions for recall evaluation)

Download the dataset from:
    https://huggingface.co/datasets/xiaowu0162/longmemeval
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Question type constants
# ---------------------------------------------------------------------------
QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
    "abstention",
}

# Map raw question_type strings to canonical labels
_TYPE_ALIASES: Dict[str, str] = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
    "multi-session": "multi-session",
}


@dataclass
class ChatTurn:
    """A single turn in a conversation session."""

    role: str  # "user" or "assistant"
    content: str
    has_answer: bool = False  # True if this turn contains the evidence


@dataclass
class ChatSession:
    """A single conversation session with a timestamp."""

    session_id: str
    date: str  # ISO date string, e.g. "2024-03-15"
    turns: List[ChatTurn] = field(default_factory=list)

    @property
    def date_as_datetime(self) -> Optional[datetime]:
        """Parse date string to datetime object."""
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y"):
            try:
                return datetime.strptime(self.date, fmt)
            except ValueError:
                continue
        return None

    def to_plain_text(self) -> str:
        """Convert session to plain text for ingestion."""
        lines = [f"[Session date: {self.date}]"]
        for turn in self.turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)


@dataclass
class LongMemEvalInstance:
    """
    A single LongMemEval evaluation instance.

    Attributes:
        question_id: Unique identifier (ends with "_abs" for abstention).
        question_type: Canonical question type label.
        is_abstention: True if the question requires abstaining (no answer).
        question: The evaluation question text.
        answer: Ground-truth answer string.
        question_date: Date when the question is asked.
        sessions: Ordered list of chat sessions (haystack).
        evidence_session_ids: Session IDs that contain the answer evidence.
    """

    question_id: str
    question_type: str
    is_abstention: bool
    question: str
    answer: str
    question_date: str
    sessions: List[ChatSession] = field(default_factory=list)
    evidence_session_ids: List[str] = field(default_factory=list)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def num_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def evidence_sessions(self) -> List[ChatSession]:
        """Return only the sessions that contain evidence."""
        id_set = set(self.evidence_session_ids)
        return [s for s in self.sessions if s.session_id in id_set]


class LongMemEvalLoader:
    """
    Loader for the LongMemEval-S dataset.

    Args:
        data_path: Path to the ``longmemeval_s.json`` file.
        max_instances: If set, only load the first N instances (useful for
            quick smoke tests).
        question_types: If set, only load instances of the specified types.
            Use ``None`` to load all types.

    Example::

        loader = LongMemEvalLoader("longmemeval_s.json")
        instances = loader.load()
        print(f"Loaded {len(instances)} instances")
    """

    def __init__(
        self,
        data_path: str,
        max_instances: Optional[int] = None,
        question_types: Optional[List[str]] = None,
    ) -> None:
        self.data_path = data_path
        self.max_instances = max_instances
        self.question_types = set(question_types) if question_types else None

    def load(self) -> List[LongMemEvalInstance]:
        """Load and parse all instances from the dataset file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"LongMemEval dataset not found at: {self.data_path}\n"
                "Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval\n"
                "  huggingface-cli download xiaowu0162/longmemeval "
                "--repo-type dataset --local-dir ./longmemeval_data"
            )

        with open(self.data_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected a JSON array in {self.data_path}, "
                f"got {type(raw_data).__name__}"
            )

        instances: List[LongMemEvalInstance] = []
        for raw in raw_data:
            instance = self._parse_instance(raw)
            if instance is None:
                continue
            if self.question_types and instance.question_type not in self.question_types:
                continue
            instances.append(instance)
            if self.max_instances and len(instances) >= self.max_instances:
                break

        return instances

    def _parse_instance(self, raw: dict) -> Optional[LongMemEvalInstance]:
        """Parse a single raw JSON object into a LongMemEvalInstance."""
        try:
            qid = str(raw["question_id"])
            is_abstention = qid.endswith("_abs")

            raw_type = raw.get("question_type", "")
            if is_abstention:
                q_type = "abstention"
            else:
                q_type = _TYPE_ALIASES.get(raw_type, raw_type)

            sessions = self._parse_sessions(
                session_ids=raw.get("haystack_session_ids", []),
                dates=raw.get("haystack_dates", []),
                session_contents=raw.get("haystack_sessions", []),
            )

            return LongMemEvalInstance(
                question_id=qid,
                question_type=q_type,
                is_abstention=is_abstention,
                question=str(raw.get("question", "")),
                answer=str(raw.get("answer", "")),
                question_date=str(raw.get("question_date", "")),
                sessions=sessions,
                evidence_session_ids=list(raw.get("answer_session_ids", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            import warnings
            warnings.warn(
                f"Skipping malformed instance (id={raw.get('question_id', '?')}): {exc}"
            )
            return None

    @staticmethod
    def _parse_sessions(
        session_ids: List[str],
        dates: List[str],
        session_contents: List[List[dict]],
    ) -> List[ChatSession]:
        """Parse raw session data into ChatSession objects."""
        sessions: List[ChatSession] = []
        for sid, date, turns_raw in zip(session_ids, dates, session_contents):
            turns = []
            for turn in turns_raw:
                turns.append(
                    ChatTurn(
                        role=str(turn.get("role", "user")),
                        content=str(turn.get("content", "")),
                        has_answer=bool(turn.get("has_answer", False)),
                    )
                )
            sessions.append(
                ChatSession(session_id=str(sid), date=str(date), turns=turns)
            )
        return sessions

    @staticmethod
    def get_download_instructions() -> str:
        """Return instructions for downloading the dataset."""
        return (
            "Download LongMemEval dataset:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli download xiaowu0162/longmemeval \\\n"
            "    --repo-type dataset --local-dir ./longmemeval_data\n\n"
            "Then pass the path to LongMemEvalLoader:\n"
            "  loader = LongMemEvalLoader('./longmemeval_data/longmemeval_s.json')\n"
        )


def get_type_distribution(instances: List[LongMemEvalInstance]) -> Dict[str, int]:
    """Count instances by question type."""
    dist: Dict[str, int] = {}
    for inst in instances:
        dist[inst.question_type] = dist.get(inst.question_type, 0) + 1
    return dict(sorted(dist.items()))
