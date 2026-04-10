"""
Time-enhanced retrieval module

Provides:
- query_time_parser: Parse time expressions in queries
- time_bonus: Calculate time match bonus
- mentioned_time_extractor: Extract event occurrence time during ingestion (Phase 2)
- recency_decay: Exponential recency-based memory decay scoring
"""

from .query_time_parser import (
    parse_query_time,
    QueryTimeInfo,
    contains_time_hint,
)

from .time_bonus import (
    compute_time_match,
    apply_time_bonus_to_results,
    TimeBonus,
    TimeBonusConfig,
)

from .mentioned_time_extractor import (
    extract_mentioned_time,
    merge_mentioned_times,
    dict_to_mentioned_time,
    MentionedTimeResult,
    validate_time_range,
)

from .recency_decay import (
    compute_decay_factor,
    compute_recency_decay,
    apply_recency_decay_to_results,
    RecencyDecayConfig,
    RecencyDecayResult,
)

__all__ = [
    # Query time parsing
    "parse_query_time",
    "QueryTimeInfo",
    "contains_time_hint",
    # Time bonus
    "compute_time_match",
    "apply_time_bonus_to_results",
    "TimeBonus",
    "TimeBonusConfig",
    # Mentioned time extraction
    "extract_mentioned_time",
    "merge_mentioned_times",
    "dict_to_mentioned_time",
    "MentionedTimeResult",
    # Time validation (Task 1.3)
    "validate_time_range",
    # Recency decay
    "compute_decay_factor",
    "compute_recency_decay",
    "apply_recency_decay_to_results",
    "RecencyDecayConfig",
    "RecencyDecayResult",
]
