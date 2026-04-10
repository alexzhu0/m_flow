"""
Comprehensive unit tests for m_flow.retrieval.time.recency_decay.

Covers:
- compute_decay_factor: mathematical correctness, boundary conditions
- compute_recency_decay: payload extraction, timestamp coercion
- apply_recency_decay_to_results: batch application, statistics
- RecencyDecayConfig: default values, skip-when-query-has-time
- _coerce_to_epoch_ms: various timestamp formats

All tests use deterministic reference_time_ms to avoid flakiness.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from m_flow.retrieval.time.recency_decay import (
    RecencyDecayConfig,
    RecencyDecayResult,
    compute_decay_factor,
    compute_recency_decay,
    apply_recency_decay_to_results,
    _coerce_to_epoch_ms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
NOW_MS = 1_700_000_000_000  # ~Nov 2023, a convenient fixed point


def _cfg(**overrides) -> RecencyDecayConfig:
    """Create an enabled config with deterministic reference time."""
    defaults = dict(
        enabled=True,
        half_life_days=30.0,
        max_boost=0.04,
        min_retention=0.05,
        reference_time_ms=NOW_MS,
        score_source="created_at",
        skip_when_query_has_time=True,
    )
    defaults.update(overrides)
    return RecencyDecayConfig(**defaults)


@dataclass
class FakeHit:
    """Minimal search-hit stub."""

    score: float
    payload: Optional[Dict[str, Any]] = None


# ===================================================================
# Tests for compute_decay_factor
# ===================================================================

class TestComputeDecayFactor:
    """Mathematical correctness of the exponential decay function."""

    def test_zero_age_returns_one(self):
        assert compute_decay_factor(0, 30.0) == 1.0

    def test_negative_age_returns_one(self):
        """Negative age (future timestamp) should clamp to 1.0."""
        assert compute_decay_factor(-1000, 30.0) == 1.0

    def test_half_life_returns_half(self):
        """At exactly one half-life, factor should be 0.5."""
        age = 30 * DAY_MS
        factor = compute_decay_factor(age, 30.0)
        assert abs(factor - 0.5) < 1e-9

    def test_two_half_lives_returns_quarter(self):
        age = 60 * DAY_MS
        factor = compute_decay_factor(age, 30.0)
        assert abs(factor - 0.25) < 1e-9

    def test_three_half_lives(self):
        age = 90 * DAY_MS
        factor = compute_decay_factor(age, 30.0)
        assert abs(factor - 0.125) < 1e-9

    def test_very_old_memory_clamps_to_min_retention(self):
        """After many half-lives, factor should not go below min_retention."""
        age = 365 * DAY_MS  # 1 year with 30-day half-life
        factor = compute_decay_factor(age, 30.0, min_retention=0.05)
        assert factor == 0.05

    def test_custom_min_retention(self):
        age = 365 * DAY_MS
        factor = compute_decay_factor(age, 30.0, min_retention=0.1)
        assert factor == 0.1

    def test_min_retention_zero(self):
        """min_retention=0 allows full decay."""
        age = 3650 * DAY_MS  # 10 years
        factor = compute_decay_factor(age, 30.0, min_retention=0.0)
        assert factor >= 0.0
        assert factor < 0.001  # should be extremely small

    def test_short_half_life(self):
        """1-day half-life: 1 day old -> 0.5."""
        age = 1 * DAY_MS
        factor = compute_decay_factor(age, 1.0)
        assert abs(factor - 0.5) < 1e-9

    def test_long_half_life(self):
        """365-day half-life: 30 days old -> still quite high."""
        age = 30 * DAY_MS
        factor = compute_decay_factor(age, 365.0)
        assert factor > 0.9  # should be ~0.944

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_decay_factor(1000, 0.0)
        with pytest.raises(ValueError, match="positive"):
            compute_decay_factor(1000, -10.0)

    def test_monotonically_decreasing(self):
        """Factor must decrease as age increases."""
        ages = [0, 1, 7, 30, 90, 180, 365]
        factors = [
            compute_decay_factor(d * DAY_MS, 30.0, min_retention=0.0)
            for d in ages
        ]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1]

    @pytest.mark.parametrize(
        "age_days,expected_approx",
        [
            (0, 1.0),
            (10, 0.7937),  # 2^(-10/30)
            (15, 0.7071),  # 2^(-0.5)
            (30, 0.5),
            (60, 0.25),
            (120, 0.0625),
        ],
    )
    def test_known_values(self, age_days, expected_approx):
        factor = compute_decay_factor(age_days * DAY_MS, 30.0, min_retention=0.0)
        assert abs(factor - expected_approx) < 0.001


# ===================================================================
# Tests for _coerce_to_epoch_ms
# ===================================================================

class TestCoerceToEpochMs:
    """Timestamp format conversion."""

    def test_none_returns_none(self):
        assert _coerce_to_epoch_ms(None) is None

    def test_milliseconds_passthrough(self):
        """Values > 1e12 are treated as milliseconds."""
        ts = 1_700_000_000_000
        assert _coerce_to_epoch_ms(ts) == ts

    def test_seconds_to_ms(self):
        """Values in the 1e9-1e12 range are treated as seconds."""
        ts_sec = 1_700_000_000
        assert _coerce_to_epoch_ms(ts_sec) == ts_sec * 1000

    def test_microseconds_to_ms(self):
        """Values > 1e15 are treated as microseconds."""
        ts_us = 1_700_000_000_000_000
        assert _coerce_to_epoch_ms(ts_us) == ts_us // 1000

    def test_float_seconds(self):
        ts = 1_700_000_000.5
        result = _coerce_to_epoch_ms(ts)
        assert result == 1_700_000_000_000  # truncated

    def test_iso_string_utc(self):
        result = _coerce_to_epoch_ms("2023-11-14T22:13:20+00:00")
        assert result is not None
        assert abs(result - 1_700_000_000_000) < 1000

    def test_iso_string_with_z(self):
        result = _coerce_to_epoch_ms("2023-11-14T22:13:20Z")
        assert result is not None

    def test_invalid_string_returns_none(self):
        assert _coerce_to_epoch_ms("not-a-date") is None

    def test_small_int_treated_as_seconds(self):
        """Very small integers are still treated as seconds."""
        result = _coerce_to_epoch_ms(1000)
        assert result == 1_000_000  # 1000 seconds in ms


# ===================================================================
# Tests for compute_recency_decay (single candidate)
# ===================================================================

class TestComputeRecencyDecay:
    """Single-candidate recency decay computation."""

    def test_disabled_config_returns_zero(self):
        cfg = RecencyDecayConfig(enabled=False)
        result = compute_recency_decay({"payload": {"created_at": NOW_MS}}, cfg)
        assert result.boost == 0.0
        assert result.decay_factor == 1.0

    def test_none_config_returns_zero(self):
        """Default config is disabled."""
        result = compute_recency_decay({"payload": {"created_at": NOW_MS}})
        assert result.boost == 0.0

    def test_fresh_memory_gets_full_boost(self):
        """Memory created 'now' should get max_boost."""
        cfg = _cfg()
        candidate = {"payload": {"created_at": NOW_MS}}
        result = compute_recency_decay(candidate, cfg)
        assert abs(result.decay_factor - 1.0) < 1e-9
        assert abs(result.boost - cfg.max_boost) < 1e-9
        assert result.age_days < 0.001

    def test_30_day_old_memory_gets_half_boost(self):
        cfg = _cfg(half_life_days=30.0, max_boost=0.04)
        ts = NOW_MS - 30 * DAY_MS
        result = compute_recency_decay({"payload": {"created_at": ts}}, cfg)
        assert abs(result.decay_factor - 0.5) < 1e-9
        assert abs(result.boost - 0.02) < 1e-9
        assert abs(result.age_days - 30.0) < 0.01

    def test_missing_timestamp_returns_missing(self):
        cfg = _cfg()
        result = compute_recency_decay({"payload": {}}, cfg)
        assert result.source_field == "missing"
        assert result.boost == 0.0

    def test_mentioned_time_source(self):
        cfg = _cfg(score_source="mentioned_time_start_ms")
        ts = NOW_MS - 10 * DAY_MS
        candidate = {"payload": {"mentioned_time_start_ms": ts}}
        result = compute_recency_decay(candidate, cfg)
        assert result.source_field == "mentioned_time_start_ms"
        assert result.boost > 0

    def test_object_with_payload_attribute(self):
        """Supports objects with .payload attribute (not just dicts)."""
        cfg = _cfg()
        hit = FakeHit(score=0.5, payload={"created_at": NOW_MS - 15 * DAY_MS})
        result = compute_recency_decay(hit, cfg)
        assert result.boost > 0
        assert result.source_field == "created_at"

    def test_iso_string_timestamp(self):
        """created_at as ISO string should be handled."""
        cfg = _cfg(reference_time_ms=1_700_000_000_000)
        candidate = {"payload": {"created_at": "2023-11-14T22:13:20Z"}}
        result = compute_recency_decay(candidate, cfg)
        # Should be ~0 days old
        assert result.age_days < 1.0
        assert result.boost > 0

    def test_future_timestamp_clamps_to_full_boost(self):
        """If memory timestamp is in the future, treat as age=0."""
        cfg = _cfg()
        future_ts = NOW_MS + 10 * DAY_MS
        result = compute_recency_decay({"payload": {"created_at": future_ts}}, cfg)
        assert result.decay_factor == 1.0


# ===================================================================
# Tests for apply_recency_decay_to_results (batch)
# ===================================================================

class TestApplyRecencyDecayToResults:
    """Batch application and statistics."""

    def _make_hits(self, ages_days: list[float], base_score: float = 0.5) -> list:
        return [
            FakeHit(
                score=base_score,
                payload={"created_at": NOW_MS - int(d * DAY_MS)},
            )
            for d in ages_days
        ]

    def test_disabled_returns_empty_stats(self):
        cfg = RecencyDecayConfig(enabled=False)
        hits = self._make_hits([0, 10, 30])
        stats = apply_recency_decay_to_results(hits, cfg)
        assert stats["decayed"] == 0
        # Scores should be unchanged
        assert all(h.score == 0.5 for h in hits)

    def test_skip_when_query_has_time(self):
        cfg = _cfg(skip_when_query_has_time=True)
        hits = self._make_hits([0, 10, 30])
        stats = apply_recency_decay_to_results(hits, cfg, query_has_time=True)
        assert stats["skipped_query_has_time"] is True
        assert stats["decayed"] == 0
        assert all(h.score == 0.5 for h in hits)

    def test_do_not_skip_when_flag_is_false(self):
        cfg = _cfg(skip_when_query_has_time=False)
        hits = self._make_hits([0, 10, 30])
        stats = apply_recency_decay_to_results(hits, cfg, query_has_time=True)
        assert stats["skipped_query_has_time"] is False
        assert stats["decayed"] == 3

    def test_fresh_memory_gets_most_boost(self):
        cfg = _cfg(max_boost=0.04)
        hits = self._make_hits([0, 30, 90])
        apply_recency_decay_to_results(hits, cfg)
        # Fresh memory should have lowest score (most boost)
        assert hits[0].score < hits[1].score < hits[2].score

    def test_scores_decrease_by_boost(self):
        cfg = _cfg(max_boost=0.04)
        hits = self._make_hits([0])
        original = hits[0].score
        apply_recency_decay_to_results(hits, cfg)
        assert hits[0].score == original - cfg.max_boost

    def test_score_floor_at_zero(self):
        """Score should not go below 0."""
        cfg = _cfg(max_boost=1.0)  # huge boost
        hits = self._make_hits([0], base_score=0.02)
        apply_recency_decay_to_results(hits, cfg)
        assert hits[0].score >= 0.0

    def test_statistics_correctness(self):
        cfg = _cfg(max_boost=0.04, half_life_days=30.0)
        hits = self._make_hits([0, 30, 60])
        stats = apply_recency_decay_to_results(hits, cfg)

        assert stats["total"] == 3
        assert stats["decayed"] == 3
        assert stats["skipped_no_timestamp"] == 0
        assert stats["max_boost"] == pytest.approx(0.04, abs=1e-6)
        assert stats["avg_age_days"] == pytest.approx(30.0, abs=0.1)
        assert 0.0 < stats["min_decay_factor"] < 1.0

    def test_missing_timestamps_counted(self):
        cfg = _cfg()
        hits = [
            FakeHit(score=0.5, payload={"created_at": NOW_MS}),
            FakeHit(score=0.5, payload={}),  # no timestamp
            FakeHit(score=0.5, payload={"created_at": NOW_MS - 10 * DAY_MS}),
        ]
        stats = apply_recency_decay_to_results(hits, cfg)
        assert stats["decayed"] == 2
        assert stats["skipped_no_timestamp"] == 1

    def test_empty_results(self):
        cfg = _cfg()
        stats = apply_recency_decay_to_results([], cfg)
        assert stats["total"] == 0
        assert stats["decayed"] == 0

    def test_hit_without_score_attribute_skipped(self):
        """Objects without .score should be silently skipped."""
        cfg = _cfg()

        class NoScore:
            payload = {"created_at": NOW_MS}

        stats = apply_recency_decay_to_results([NoScore()], cfg)
        assert stats["decayed"] == 0

    def test_ordering_preserved_with_different_ages(self):
        """Recency decay should re-rank results by freshness."""
        cfg = _cfg(max_boost=0.1)
        # All start with same score, different ages
        hits = self._make_hits([90, 0, 30], base_score=0.5)
        apply_recency_decay_to_results(hits, cfg)
        # hits[1] (0 days) should have lowest score
        # hits[2] (30 days) in the middle
        # hits[0] (90 days) highest score
        assert hits[1].score < hits[2].score < hits[0].score


# ===================================================================
# Tests for RecencyDecayConfig defaults
# ===================================================================

class TestRecencyDecayConfigDefaults:
    """Verify sensible defaults."""

    def test_disabled_by_default(self):
        cfg = RecencyDecayConfig()
        assert cfg.enabled is False

    def test_default_half_life(self):
        cfg = RecencyDecayConfig()
        assert cfg.half_life_days == 30.0

    def test_default_max_boost(self):
        cfg = RecencyDecayConfig()
        assert cfg.max_boost == 0.04

    def test_default_min_retention(self):
        cfg = RecencyDecayConfig()
        assert cfg.min_retention == 0.05

    def test_default_skip_when_query_has_time(self):
        cfg = RecencyDecayConfig()
        assert cfg.skip_when_query_has_time is True

    def test_default_score_source(self):
        cfg = RecencyDecayConfig()
        assert cfg.score_source == "created_at"


# ===================================================================
# Integration-style tests
# ===================================================================

class TestRecencyDecayIntegration:
    """End-to-end scenarios combining multiple components."""

    def test_realistic_search_scenario(self):
        """Simulate a realistic search with mixed-age results."""
        cfg = _cfg(half_life_days=7.0, max_boost=0.05)

        hits = [
            FakeHit(0.30, {"created_at": NOW_MS - 1 * HOUR_MS}),   # 1 hour ago
            FakeHit(0.28, {"created_at": NOW_MS - 3 * DAY_MS}),    # 3 days ago
            FakeHit(0.25, {"created_at": NOW_MS - 14 * DAY_MS}),   # 2 weeks ago
            FakeHit(0.35, {"created_at": NOW_MS - 60 * DAY_MS}),   # 2 months ago
            FakeHit(0.32, {"created_at": NOW_MS - 365 * DAY_MS}),  # 1 year ago
        ]

        stats = apply_recency_decay_to_results(hits, cfg)

        assert stats["decayed"] == 5
        # 1-hour-old memory should get nearly full boost
        assert hits[0].score < 0.30
        assert hits[0].score == pytest.approx(0.30 - 0.05, abs=0.001)
        # 1-year-old memory should get almost no boost (min_retention * max_boost)
        assert hits[4].score > 0.30

    def test_config_with_custom_half_life_changes_ranking(self):
        """Different half-lives produce different rankings."""
        hits_short = [
            FakeHit(0.5, {"created_at": NOW_MS - 7 * DAY_MS}),
            FakeHit(0.5, {"created_at": NOW_MS - 30 * DAY_MS}),
        ]
        hits_long = [
            FakeHit(0.5, {"created_at": NOW_MS - 7 * DAY_MS}),
            FakeHit(0.5, {"created_at": NOW_MS - 30 * DAY_MS}),
        ]

        cfg_short = _cfg(half_life_days=3.0, max_boost=0.1)
        cfg_long = _cfg(half_life_days=365.0, max_boost=0.1)

        apply_recency_decay_to_results(hits_short, cfg_short)
        apply_recency_decay_to_results(hits_long, cfg_long)

        # With short half-life, the gap between 7-day and 30-day should be larger
        gap_short = hits_short[1].score - hits_short[0].score
        gap_long = hits_long[1].score - hits_long[0].score
        assert gap_short > gap_long
