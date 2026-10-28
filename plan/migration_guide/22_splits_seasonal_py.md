# Migration Map: `src/preprocessing_core/splits/seasonal.py`

## Purpose

Implement the built-in `seasonal_block` split policy described in `MINIMAL_PREPROCESSING_CORE_PLAN.md`.

This is the only built-in split in v1 that depends on `dataset.timestamps`. It should stay generic: calendar year/month in, split indices out.

## Current repo reference

The nearest current implementation is `SeasonStratifiedBlockSplit` in:

- `CrossExperimentalAIDQM/src/anldq/datasets/process/split_time.py:340-497`

Useful behavior to preserve from current code:

- require timestamps and reject missing timestamps;
- convert timestamps to calendar datetimes;
- map months into named seasons;
- stratify by `(year, season)`;
- split each stratum into contiguous blocks;
- assign whole blocks to train/val/test using seeded randomness;
- optionally trim a `gap` from both ends of each chosen block;
- return indices in final chronological order.

Behavior to simplify or modernize for the new public core:

- return one split-index dict for all three splits, not one object per split;
- use `np.random.default_rng(...)`, not `np.random.RandomState(...)`;
- do not use Python `hash()` for seeds;
- validate non-empty train/val/test at the framework boundary;
- keep provenance small and pipeline-owned.

## Scope

Owns:

- calendar conversion from timestamps
- season mapping validation
- contiguous block construction within each `(year, season)` stratum
- seeded whole-block assignment to train/val/test
- optional block-edge gap trimming
- returning plain split-index dicts

Does not own:

- bundle metadata assembly
- dataset slicing
- split-index validation beyond local sanity checks
- timestamp monotonicity repair

## Public shape

```python
"""Seasonal block split policy for timestamped datasets."""

from collections.abc import Mapping, Sequence
import numpy as np

from preprocessing_core.datasets import BaseDataset


def seasonal_block_split(
    dataset: BaseDataset,
    *,
    ratios: tuple[float, float, float],
    blocks_per_stratum: int = 12,
    gap: int = 0,
    seed: int = 42,
    seasons: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    ...
```

## Minimal algorithm

```python
def seasonal_block_split(dataset: BaseDataset, **params) -> dict[str, np.ndarray]:
    if dataset.timestamps is None:
        raise ConfigError("seasonal_block requires dataset.timestamps")

    timestamps = np.asarray(dataset.timestamps)
    years, months = extract_calendar_fields(timestamps)
    season_map = normalize_season_map(params.get("seasons"))
    month_to_season = invert_season_map(season_map)
    season_ids = np.asarray([month_to_season[int(month)] for month in months], dtype=object)

    selected = {"train": [], "val": [], "test": []}
    for year in np.unique(years):
        for season_name in season_map:
            group = np.flatnonzero((years == year) & (season_ids == season_name))
            if group.size == 0:
                continue

            blocks = make_contiguous_blocks(group, blocks_per_stratum)
            assigned = assign_blocks_to_splits(blocks=blocks, ratios=ratios, seed=seed, year=int(year), season=season_name)
            for split_name, split_blocks in assigned.items():
                for block in split_blocks:
                    trimmed = trim_block_gap(block, gap)
                    if trimmed.size:
                        selected[split_name].append(trimmed)

    return {
        split_name: merge_and_sort_blocks(selected[split_name])
        for split_name in ("train", "val", "test")
    }
```

## Helper boundaries

Keep helpers small and local to this file unless reused elsewhere.

Suggested helpers:

- `extract_calendar_fields(timestamps) -> tuple[np.ndarray, np.ndarray]`
- `normalize_season_map(seasons) -> dict[str, tuple[int, ...]]`
- `invert_season_map(season_map) -> dict[int, str]`
- `make_contiguous_blocks(group, blocks_per_stratum) -> list[np.ndarray]`
- `allocate_block_counts(n_blocks, ratios) -> tuple[int, int, int]`
- `assign_blocks_to_splits(...) -> dict[str, list[np.ndarray]]`
- `trim_block_gap(block, gap) -> np.ndarray`
- `merge_and_sort_blocks(blocks) -> np.ndarray`

## Important implementation notes

### Timestamp conversion

Mirror current behavior conceptually:

- convert to calendar datetimes;
- reject timestamps that cannot be converted;
- extract year and month only.

Using pandas for conversion is acceptable in v1 if the package already depends on it for the CSV loader. Keep the conversion in one helper so it is easy to replace later if desired.

### Season map validation

Preserve current guardrails:

- every month must appear exactly once;
- months must be integers in `[1, 12]`;
- duplicate month ownership is invalid.

Default season map may remain the current one unless the plan is updated:

```python
{
    "summer": (12, 1, 2),
    "autumn": (3, 4, 5),
    "winter": (6, 7, 8),
    "spring": (9, 10, 11),
}
```

The names are labels only; correctness comes from complete month coverage.

### Block assignment and randomness

Keep the current useful idea:

- assignment should be reproducible per `(year, season)` stratum;
- each stratum should use a deterministic seed derived from `(seed, year, season)`.

But simplify the implementation:

- do not use `hashlib` if a small deterministic helper is enough;
- do not use Python `hash()`;
- do use `np.random.default_rng(local_seed)`.

A reasonable approach is to derive `local_seed` from a stable string encoding of `(seed, year, season)` via a small digest helper.

### Empty split behavior

Current code can silently return empty selections for some strata or whole outputs. The new core must not silently accept an empty final split. That final validation belongs to the shared split-index validator, not duplicated here.

Local helper behavior should still be conservative:

- trimming a gap that consumes a whole block should drop that block;
- if a stratum cannot contribute to a split, that is acceptable locally;
- the final merged split arrays are validated globally afterward.

### Chronological output

Return sorted row indices per split. Even though block assignment is randomized, the final indices within each split should be in dataset row order.

## What not to carry over

Do not carry over the current class-per-split API:

- no dataclass with `.split` field;
- no `.with_split(...)` method;
- no `get_group_lengths()` method.

If future diagnostics need dropped-row counts or block counts, compute that in a pipeline helper for bundle metadata rather than expanding the split policy return type.

## Split metadata expectations

This split should not return metadata itself. The pipeline may compute small provenance after the fact, for example:

```python
{
    "dropped_rows": {
        "gap_within_blocks": ...,
    }
}
```

Keep this optional and small. Do not add a richer split result type for v1.

## Tests

Add tests for:

- missing timestamps raises clear error
- non-convertible timestamps raises clear error
- invalid season map raises clear error
- month coverage must be exactly complete
- output contains `train`, `val`, `test`
- output indices are integer arrays within range
- output indices are sorted within each split
- reproducibility for a fixed seed
- different seeds can change assignment
- `gap` trims block edges
- large `gap` can eliminate individual blocks, but global validator rejects empty final splits
- no overlap across train/val/test after merge
- custom season map works

## Human readability

- Keep the public function short.
- Keep the helper names literal.
- Prefer a top-to-bottom flow a reviewer can read in one pass.
- Do not hide the core logic in an abstract block allocator class.
