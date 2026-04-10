# Recency Decay Scoring

M-Flow's retrieval pipeline already supports **explicit temporal matching**
via the `time_bonus` module — when a user asks "What happened in March
2024?", candidates whose timestamps overlap that range receive a score
boost.  However, many queries carry an *implicit* recency preference
without mentioning a specific date:

> "What did we discuss about the product roadmap?"

In such cases, a conversation from last week is almost certainly more
relevant than one from two years ago, yet the existing pipeline treats
them equally.  **Recency decay** fills this gap by giving fresher
memories a small, configurable score boost that decays exponentially
over time.

---

## Academic Foundations

The decay function is grounded in well-established research:

| Source | Formula | Key Idea |
|--------|---------|----------|
| Ebbinghaus (1885) | R = exp(-t/S) | Forgetting curve: memory retention decays exponentially |
| Generative Agents (Park et al., UIST 2023) | recency = γ^hours | Decay factor applied to agent memory retrieval scoring |
| SynapticRAG (Xu et al., ACL 2025 Findings) | f(x) = exp(-x/τ) | Temporal scoring with configurable time constant |
| MemoryBank (Zhong et al., AAAI 2024) | Ebbinghaus-based update | Dynamic memory retention/forgetting mechanism |

M-Flow's implementation uses a **half-life formulation**:

```
decay_factor = exp(-age × ln(2) / half_life)
```

This guarantees that `decay_factor = 0.5` when `age = half_life`,
providing an intuitive configuration knob.

---

## Quick Start

Set the following environment variables to enable recency decay:

```bash
# Enable recency decay (disabled by default)
export MFLOW_RECENCY_DECAY_ENABLED=true

# Half-life in days (default: 30)
# Shorter = stronger preference for recent memories
export MFLOW_RECENCY_DECAY_HALF_LIFE_DAYS=30

# Maximum score boost for the freshest memories (default: 0.04)
export MFLOW_RECENCY_DECAY_MAX_BOOST=0.04
```

No code changes are required.  When enabled, recency decay is
automatically applied after the existing time-bonus stage in the
retrieval pipeline.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Master switch |
| `half_life_days` | float | `30.0` | Days until decay factor reaches 0.5 |
| `max_boost` | float | `0.04` | Maximum score reduction for fresh memories |
| `min_retention` | float | `0.05` | Floor for decay factor (prevents total suppression) |
| `score_source` | str | `"created_at"` | Timestamp field to use (`created_at` or `mentioned_time_start_ms`) |
| `skip_when_query_has_time` | bool | `True` | Skip decay when query has explicit time reference |
| `reference_time_ms` | int \| None | `None` | Override "now" for testing/evaluation |

---

## How It Works

### Decay Curve

```
decay_factor
  1.0 ┤●
      │ ╲
  0.8 ┤  ╲
      │   ╲
  0.6 ┤    ╲
      │     ╲
  0.5 ┤------●---- (half-life)
      │       ╲
  0.4 ┤        ╲
      │         ╲
  0.25┤----------●---- (2× half-life)
      │           ╲
  0.1 ┤            ╲___
  0.05┤-----------------●---- (min_retention floor)
      └─────┬─────┬─────┬────→ age (days)
            30    60    90
```

### Integration with Existing Pipeline

```
Query → Parse Time → Vector Search → Time Bonus → Recency Decay → Sort → Top-K
                                      ↑               ↑
                                      │               │
                              Explicit time       Implicit freshness
                              ("March 2024")      (no time mentioned)
```

When the query contains an explicit time reference (detected by the
time parser), recency decay is **skipped by default**
(`skip_when_query_has_time=True`).  This prevents the two mechanisms
from conflicting — explicit time matching takes priority.

### Score Impact

The actual score modification is:

```
new_score = current_score - (decay_factor × max_boost)
```

Since lower scores rank higher in M-Flow, this means:
- **Fresh memories** (decay_factor ≈ 1.0): get the full `max_boost` reduction
- **Old memories** (decay_factor → min_retention): get almost no boost
- **Very old memories**: still retain `min_retention × max_boost` to avoid
  completely suppressing highly relevant ancient content

---

## Choosing the Right Half-Life

| Use Case | Recommended Half-Life | Rationale |
|----------|----------------------|-----------|
| Customer support chat | 7 days | Recent issues are most relevant |
| Personal assistant | 30 days (default) | Balance between recent and historical |
| Research knowledge base | 90–180 days | Papers and findings stay relevant longer |
| Legal/compliance archive | 365+ days | Historical records remain important |

---

## Programmatic Usage

```python
from m_flow.retrieval.time.recency_decay import (
    RecencyDecayConfig,
    apply_recency_decay_to_results,
    compute_decay_factor,
)

# Check decay factor for a 15-day-old memory with 30-day half-life
factor = compute_decay_factor(
    age_ms=15 * 86_400_000,
    half_life_days=30.0,
)
print(f"Decay factor: {factor:.4f}")  # ~0.7071

# Apply to search results
config = RecencyDecayConfig(
    enabled=True,
    half_life_days=14.0,
    max_boost=0.05,
)
stats = apply_recency_decay_to_results(results, config)
print(f"Decayed {stats['decayed']} results, avg boost: {stats['avg_boost']:.4f}")
```

---

## Testing

Run the dedicated test suite:

```bash
python -m pytest m_flow/tests/unit/retrieval/test_recency_decay.py -v
```

The test suite includes 55 test cases covering:
- Mathematical correctness of the decay function (boundary values, known points)
- Timestamp format coercion (ms, seconds, microseconds, ISO strings)
- Single-candidate and batch processing
- Configuration defaults and edge cases
- Realistic search scenarios with mixed-age results
