"""
Recency decay scoring module.

Implements time-based memory decay inspired by:
- Ebbinghaus forgetting curve: R = exp(-t/S)
- Generative Agents (Park et al. 2023): recency = γ^hours_since_access
- SynapticRAG (ACL 2025 Findings): f(x) = exp(-x/τ)

The core idea is that more recent memories should be ranked higher
when no explicit time reference is present in the query.  This
complements the existing ``time_bonus`` module which handles
*explicit* temporal matching.  Recency decay handles the *implicit*
preference for fresh information.

Design principles:
- Pure functions with no side effects for easy testing
- Configurable half-life to suit different domains
- Default OFF to avoid breaking existing behaviour
- Integrates additively with the existing time-bonus pipeline
- Lower score = better ranking (consistent with m_flow convention)

References:
    [1] Ebbinghaus, H. (1885). Memory: A Contribution to Experimental
        Psychology.
    [2] Park, J. S. et al. (2023). Generative Agents: Interactive
        Simulacra of Human Behavior. UIST 2023.
    [3] Xu, Z. et al. (2024). SynapticRAG: Improving RAG with
        Synaptic-Inspired Temporal Scoring. ACL 2025 Findings.
        arXiv:2410.13553.
    [4] Zhong, W. et al. (2024). MemoryBank: Enhancing Large Language
        Models with Long-Term Memory. AAAI 2024.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RecencyDecayConfig:
    """Configuration for recency-based memory decay.

    Attributes
    ----------
    enabled : bool
        Master switch.  When ``False`` the module is a no-op.
    half_life_days : float
        The number of days after which a memory's recency score drops
        to 0.5.  Shorter values make the system prefer very recent
        memories; longer values are more tolerant of older content.
        Default is 30 days.
    max_boost : float
        Maximum score reduction (boost) applied to the freshest
        memories.  This caps the influence of recency so it cannot
        overwhelm semantic relevance.  Default is 0.04.
    min_retention : float
        Floor for the decay multiplier.  Even very old memories will
        retain at least this fraction of ``max_boost``.  Prevents
        ancient but highly relevant memories from being completely
        suppressed.  Default is 0.05 (5 %).
    reference_time_ms : int | None
        If set, use this as "now" instead of the wall clock.  Useful
        for deterministic testing and offline evaluation.
    score_source : str
        Which timestamp field to use from the candidate payload.
        ``"created_at"`` (default) uses the memory ingestion time.
        ``"mentioned_time_start_ms"`` uses the event occurrence time.
    skip_when_query_has_time : bool
        When ``True`` (default), recency decay is *not* applied if the
        query already contains an explicit time reference (handled by
        ``time_bonus`` instead).  Set to ``False`` to always apply.
    """

    enabled: bool = False
    half_life_days: float = 30.0
    max_boost: float = 0.04
    min_retention: float = 0.05
    reference_time_ms: Optional[int] = None
    score_source: str = "created_at"
    skip_when_query_has_time: bool = True


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RecencyDecayResult:
    """Result of a single recency-decay computation.

    Attributes
    ----------
    decay_factor : float
        The raw decay multiplier in ``[min_retention, 1.0]``.
    boost : float
        The actual score reduction to apply (``decay_factor * max_boost``).
    age_days : float
        Age of the memory in days.
    source_field : str
        Which payload field was used (``"created_at"`` or
        ``"mentioned_time_start_ms"``).
    """

    decay_factor: float = 1.0
    boost: float = 0.0
    age_days: float = 0.0
    source_field: str = "none"


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------

def _now_ms(config: RecencyDecayConfig) -> int:
    """Return the current time in epoch milliseconds."""
    if config.reference_time_ms is not None:
        return config.reference_time_ms
    return int(time.time() * 1000)


def compute_decay_factor(
    age_ms: int,
    half_life_days: float,
    min_retention: float = 0.05,
) -> float:
    """Compute the exponential decay factor for a given age.

    Uses the formula::

        factor = exp(-age / (half_life * ln(2)))

    which guarantees that ``factor == 0.5`` when ``age == half_life``.

    Parameters
    ----------
    age_ms : int
        Age of the memory in **milliseconds**.  Must be >= 0.
    half_life_days : float
        Half-life in days.  Must be > 0.
    min_retention : float
        Floor value for the returned factor.

    Returns
    -------
    float
        Decay factor in ``[min_retention, 1.0]``.

    Examples
    --------
    >>> compute_decay_factor(0, 30.0)
    1.0
    >>> round(compute_decay_factor(30 * 86400 * 1000, 30.0), 2)
    0.5
    >>> round(compute_decay_factor(60 * 86400 * 1000, 30.0), 2)
    0.25
    """
    if age_ms <= 0:
        return 1.0
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    half_life_ms = half_life_days * 86_400_000  # days -> ms
    # exp(-age / (half_life / ln2)) = exp(-age * ln2 / half_life)
    exponent = -(age_ms * math.log(2)) / half_life_ms
    raw = math.exp(exponent)
    return max(min_retention, min(1.0, raw))


def compute_recency_decay(
    candidate: Any,
    config: Optional[RecencyDecayConfig] = None,
) -> RecencyDecayResult:
    """Compute recency decay for a single search candidate.

    Parameters
    ----------
    candidate
        A search result object or dict.  Must expose a ``payload``
        dict (or be a dict itself) containing a timestamp field
        specified by ``config.score_source``.
    config
        Decay configuration.  Defaults to a disabled config.

    Returns
    -------
    RecencyDecayResult
    """
    if config is None:
        config = RecencyDecayConfig()

    if not config.enabled:
        return RecencyDecayResult()

    # --- Extract payload ------------------------------------------------
    payload: Dict[str, Any] = {}
    if isinstance(candidate, dict):
        payload = candidate.get("payload", candidate)
    elif hasattr(candidate, "payload") and candidate.payload is not None:
        payload = candidate.payload if isinstance(candidate.payload, dict) else {}

    # --- Extract timestamp ----------------------------------------------
    ts_ms: Optional[int] = None
    source_field = config.score_source

    if source_field == "created_at":
        raw = payload.get("created_at")
        if raw is not None:
            ts_ms = _coerce_to_epoch_ms(raw)
    elif source_field == "mentioned_time_start_ms":
        raw = payload.get("mentioned_time_start_ms")
        if raw is not None:
            ts_ms = int(raw)
    else:
        # Fallback: try the literal field name
        raw = payload.get(source_field)
        if raw is not None:
            ts_ms = _coerce_to_epoch_ms(raw)

    if ts_ms is None:
        return RecencyDecayResult(source_field="missing")

    # --- Compute age and decay ------------------------------------------
    now = _now_ms(config)
    age_ms = max(0, now - ts_ms)
    age_days = age_ms / 86_400_000

    factor = compute_decay_factor(
        age_ms,
        config.half_life_days,
        config.min_retention,
    )
    boost = factor * config.max_boost

    return RecencyDecayResult(
        decay_factor=factor,
        boost=boost,
        age_days=age_days,
        source_field=source_field,
    )


# ---------------------------------------------------------------------------
# Batch application
# ---------------------------------------------------------------------------

def apply_recency_decay_to_results(
    results: List[Any],
    config: Optional[RecencyDecayConfig] = None,
    query_has_time: bool = False,
) -> Dict[str, Any]:
    """Apply recency decay to a list of search results.

    Directly mutates the ``score`` attribute of each result object
    (lower score = better ranking, consistent with m_flow convention).

    Parameters
    ----------
    results
        List of search-hit objects with a ``score`` attribute.
    config
        Decay configuration.
    query_has_time
        Whether the current query contains an explicit time reference.
        When ``True`` and ``config.skip_when_query_has_time`` is also
        ``True``, decay is skipped entirely.

    Returns
    -------
    dict
        Statistics about the decay application.
    """
    if config is None:
        config = RecencyDecayConfig()

    stats: Dict[str, Any] = {
        "total": len(results),
        "decayed": 0,
        "skipped_no_timestamp": 0,
        "skipped_query_has_time": False,
        "avg_decay_factor": 0.0,
        "avg_boost": 0.0,
        "max_boost": 0.0,
        "min_decay_factor": 1.0,
        "avg_age_days": 0.0,
    }

    if not config.enabled:
        return stats

    if query_has_time and config.skip_when_query_has_time:
        stats["skipped_query_has_time"] = True
        return stats

    total_factor = 0.0
    total_boost = 0.0
    total_age = 0.0

    for r in results:
        current_score = getattr(r, "score", None)
        if current_score is None:
            continue

        decay = compute_recency_decay(r, config)

        if decay.source_field == "missing":
            stats["skipped_no_timestamp"] += 1
            continue

        # Apply boost (reduce score — lower is better)
        new_score = max(0.0, current_score - decay.boost)
        r.score = new_score

        stats["decayed"] += 1
        total_factor += decay.decay_factor
        total_boost += decay.boost
        total_age += decay.age_days
        stats["max_boost"] = max(stats["max_boost"], decay.boost)
        stats["min_decay_factor"] = min(
            stats["min_decay_factor"], decay.decay_factor
        )

    if stats["decayed"] > 0:
        stats["avg_decay_factor"] = total_factor / stats["decayed"]
        stats["avg_boost"] = total_boost / stats["decayed"]
        stats["avg_age_days"] = total_age / stats["decayed"]

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_to_epoch_ms(value: Any) -> Optional[int]:
    """Best-effort conversion of a timestamp value to epoch milliseconds.

    Handles:
    - int/float already in ms (> 1e12)
    - int/float in seconds (< 1e12)
    - ISO-8601 date strings (via ``datetime.fromisoformat``)
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        v = int(value)
        if v > 1e15:
            # Likely microseconds
            return v // 1000
        if v > 1e12:
            # Already milliseconds
            return v
        if v > 1e9:
            # Seconds
            return v * 1000
        # Too small — treat as seconds anyway
        return v * 1000

    if isinstance(value, str):
        try:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return None

    return None
