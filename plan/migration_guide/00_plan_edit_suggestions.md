# Suggested Edits to `MINIMAL_PREPROCESSING_CORE_PLAN.md`

## Verdict

The plan is good and should remain a public mini-framework plan. The framework rules are useful because they reduce decision burden for downstream users and plugin authors.

The plan does **not** need to be simplified into a small utility package. It should be updated to make the framework boundaries more explicit:

- what the framework decides for users;
- what users configure;
- what plugin authors implement;
- what `PreprocessingPipeline` delegates;
- how logging works;
- how errors are categorized and wrapped.

No edits have been made to the plan yet. These are recommended changes only.

---

## Edit 1: Add a short “framework rules” section near §2

### Why

The plan currently says YAML schedules work and Python supplies operations, but it would help to explicitly state what decisions the public framework owns. Since the goal is a public framework, users should not need to re-decide pipeline semantics.

### Suggested text

Add after §2.5:

```markdown
### 2.6 Framework-owned rules

This package is intentionally a small public framework, not a loose collection of utilities. The framework owns the rules that must be consistent across downstream users:

- canonical data shape is always `(T, C, F)`;
- raw data is loaded exactly once per `build()` call;
- pre-split transforms always run once on the full dataset and never use `apply` or `fit`;
- split policies return indices only; the core materializes split datasets;
- post-split transforms run in deterministic `train -> val -> test` order;
- train-fitted state is scoped by configured step and must be serializable;
- row/channel alignment is performed only through core selection helpers;
- config validation rejects typos and unsafe combinations early;
- bundle metadata records provenance, final shapes, dropped rows, and fitted state.

Users configure these rules but do not redefine them. Plugin authors provide loaders, transforms, and split policies that obey these contracts.
```

---

## Edit 2: Add a delegation boundary table to §8

### Why

The current pipeline section says `PreprocessingPipeline` is thin, but an implementation agent could still put too much code there. A table will make the boundary explicit.

### Suggested text

Add under §8.1:

```markdown
#### Delegation boundaries

`PreprocessingPipeline` sequences work and owns build-time state. It must delegate details as follows:

| Responsibility | Owner |
| --- | --- |
| YAML parsing and schema validation | `config.py` |
| callable lookup and duplicate registration rules | `registry.py` |
| transform scheduling and `apply`/`fit` validation | small helpers in `pipeline.py` or `validation.py` |
| row/channel slicing and alignment | `selection.py` |
| dataset and split-index validation | `validation.py` / `BaseDataset.validate()` |
| loader-specific file parsing | `loaders/*` or external plugin loaders |
| transform-specific data changes | `transforms/*` or external plugin transforms |
| split index calculation | `splits/*` or external plugin policies |
| bundle metadata assembly | `make_bundle_metadata()` helper, not inline pipeline code |
| fitted-state serialization checks | validation helper, used by pipeline before metadata/persistence |
| logging policy | pipeline logs stage boundaries; helpers do not spam logs |

If a method on `PreprocessingPipeline` grows beyond sequencing and context wrapping, extract the decision into a helper function.
```

---

## Edit 3: Add a logging policy section after §13

### Why

The current plan is strong on errors but under-specified on logging. A public framework should have predictable, quiet, useful logs.

### Suggested text

Add new §13.1:

```markdown
### 13.1 Logging policy

Use the standard `logging` module. Library code must never use `print()`.

The package logs stage-level progress and important decisions only:

- `INFO` when a pipeline build starts and finishes;
- `INFO` for selected loader, split policy, and final split sizes;
- `DEBUG` before/after each configured transform, including stage, transform name, split name when relevant, and resulting shape;
- `DEBUG` for fitted-state keys created or reused;
- `WARNING` only for recoverable but suspicious behavior that the user should know about;
- no per-row, per-channel, or inner-loop logs.

Errors should normally be raised, not logged and swallowed. If a caught exception is wrapped with context, use exception chaining (`raise ... from exc`). Avoid duplicate logging at every layer; the top-level caller or CLI can decide whether to log stack traces.
```

---

## Edit 4: Clarify custom loader decomposition in §6.1 / §17

### Why

The plan correctly moves experiment-specific logic into private loaders, but that can create new monoliths. The public plan should warn plugin authors not to make giant loader files.

### Suggested text

Add to §6.1 after custom-loader discussion:

```markdown
Custom loaders may be complex, but they should still be decomposed. A loader should orchestrate source-specific assembly; it should not become an unstructured replacement for the old dataset subclasses. For complex experiments, split private loader code into focused helpers such as readers, normalizers, quality filters, channel selectors, and metadata assemblers. The public core cannot enforce this inside private plugins, but the extension docs should recommend it strongly.
```

---

## Edit 5: Make exception policy explicit in §13

### Why

The plan mentions `ConfigError` and `PipelineError`, but the policy should be explicit and small.

### Suggested text

Add near start of §13:

```markdown
Use a very small custom exception set:

- `ConfigError`: invalid YAML, invalid schema, unknown registered name, invalid `apply`/`fit`, invalid split config, or any error that should be fixed by editing configuration.
- `PipelineError`: a registered callable failed during execution, returned the wrong type, corrupted orchestrator-owned fields, or produced invalid output.

Lower-level helpers may raise built-in exceptions such as `ValueError`, `KeyError`, `FileNotFoundError`, or `ImportError`. Boundary code should wrap them only when doing so adds useful stage/config context.

Do not build a deep custom exception hierarchy in v1.
```

---

## Edit 6: Add explicit “framework defaults” to YAML/config section

### Why

Downstream users should not need to decide defaults repeatedly. The plan already has many defaults; list them in one place.

### Suggested text

Add under §7:

```markdown
#### Framework defaults

Unless explicitly configured:

- `pre_split` is an empty list;
- `post_split` is an empty list;
- post-split `apply` defaults to all splits: `train`, `val`, `test`;
- `fit` is absent by default;
- split `gap` defaults to `0` for built-in policies;
- `metadata` defaults to `{}`;
- `state_key` defaults to `{stage}:{position}:{name}`;
- labels and channel labels default to `None` when the loader does not provide them.

Defaults must be applied in config parsing, not scattered throughout transforms.
```

---

## Edit 7: Add a “public transform author checklist” to §12

### Why

This reduces mistakes by future agents or plugin authors.

### Suggested text

```markdown
#### Transform author checklist

Every transform must:

- accept and return `BaseDataset`;
- accept `state` and `fit` keyword arguments even if unused;
- avoid touching `split`, `data_tag`, or `preprocess_steps`;
- call `select_time_rows` for row filtering;
- call `select_channels` for channel filtering;
- update `feature_names` when slicing the feature axis;
- call or rely on pipeline validation after mutation;
- store only serializable fitted state;
- raise clear errors rather than silently falling back.
```

---

## Edit 8: Add module-level docstring expectations to §10 or §15

### Why

A less capable agent will otherwise skip docs. The style guide asks for human-readable public APIs and extension points.

### Suggested text

```markdown
Every public module in `preprocessing_core` must start with a short module docstring explaining its role in the framework. Public dataclasses, loader/transform/split contracts, and exported functions must have concise docstrings. Comments should explain framework rules and non-obvious edge cases, not restate obvious code.
```

---

## Edit 9: Clarify `allow_empty` is not v1

### Why

The plan says empty transforms fail unless `allow_empty: true` is introduced later. Make it clear this is not part of v1, so agents do not implement it prematurely.

### Suggested text

```markdown
`allow_empty` is not a v1 configuration field. In v1, any transform that leaves an applicable split empty raises `PipelineError`. If a future use case needs empty splits, add `allow_empty` with config validation, docs, and tests in a later version.
```

---

## Edit 10: Add non-goal against inline large helpers in `pipeline.py`

### Why

This is the concrete anti-bloat guardrail.

### Suggested text

Add to §17 non-goals:

```markdown
- No large inline implementation blocks inside `PreprocessingPipeline`; complex validation, metadata construction, selection, state serialization, and operation-specific behavior must live in focused helper functions or modules.
```
