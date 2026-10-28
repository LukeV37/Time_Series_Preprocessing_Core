# New Repo Style Guide

This guide defines the preferred Python style and repository architecture for future repos, using lessons from these reference repositories:

- `atom/`
- `salt/`
- `training-dataset-dumper/`

The goal is not to copy any one repo exactly. The goal is to combine their best patterns into a style that is:

- easy for humans to read
- easy for agents to extend safely
- architecturally consistent
- configuration-friendly
- robust in the face of errors

This guide focuses on Python, especially OOP structure, repository layout, and maintainable architecture.

---

## Core principles

### 1. Optimize for human readability
Code should be understandable by a new engineer or future agent without requiring hidden context.

Prefer:

- descriptive names
- explicit control flow
- short, direct methods
- comments that explain intent
- modules with one clear purpose

Avoid:

- clever abstractions with unclear value
- compressed logic that saves lines but hurts clarity
- comments that restate the code without adding meaning
- vague names like `data`, `util`, `handler`, or `process` unless the scope is truly generic

### 2. Keep architecture explicit
The repo should make it obvious where functionality belongs.

Use package and module boundaries to separate concerns such as:

- data access
- preprocessing or transformation
- models or domain logic
- inference or runtime execution
- orchestration or application entrypoints
- utilities shared across packages

A new contributor should be able to infer the architecture from the directory structure.

### 3. Prefer composition over deep inheritance
All three reference repos work best when they compose small units instead of building deep inheritance trees.

Prefer:

- focused classes
- dataclasses for configuration-heavy objects
- small base interfaces only when they simplify extension
- explicit assembly in pipelines, factories, or coordinators

Avoid:

- inheritance chains that hide behavior
- base classes that contain unrelated default behavior
- framework-style indirection unless it clearly reduces repetition

### 4. Make configuration flexible but controlled
Configuration should make the repo adaptable without making behavior mysterious.

Prefer:

- typed config objects or clearly structured dictionaries
- validation near configuration boundaries
- documented defaults
- config-driven composition of workflows and components

Avoid:

- silent fallback behavior that changes results unexpectedly
- large mutable config blobs passed everywhere
- configuration keys with unclear meaning or inconsistent naming

### 5. Fail clearly and early
Errors should help the reader understand what went wrong and what to do next.

Prefer:

- explicit exception types
- actionable messages
- validation at boundaries
- logging with enough context to debug operational failures

Avoid:

- generic `Exception` unless there is a strong reason
- swallowed exceptions
- broad exception handling without clear justification
- `assert` for user-facing validation

---

## What to borrow from the reference repos

### From `atom`
Use these ideas:

- clear package-level architecture
- separation of pipeline stages by responsibility
- explicit dependency flow between modules
- practical, understandable APIs
- readable orchestration with visible data flow

Do not copy blindly:

- large operational modules that gather too many responsibilities
- inconsistent error-handling style between library code and scripts

### From `salt`
Use these ideas:

- strong package boundaries
- good type hints and docstrings
- reusable, focused model or task components
- strong test culture
- clear tooling in `pyproject.toml`
- fail-fast validation with useful errors

Do not copy blindly:

- very large coordinator classes unless they are truly necessary
- heavy internal complexity when a simpler structure would work

### From `training-dataset-dumper`
Use these ideas:

- small plugin or block classes
- dataclass-based configuration objects
- minimal base interfaces
- factory-driven assembly when many variants must be supported

Do not copy blindly:

- uneven formatting or message style
- broad procedural orchestration modules that dilute the clean block model

---

## Recommended repository layout

A new repo should make the architecture visible at the top level. One good default layout is:

```text
repo/
├── README.md
├── pyproject.toml
├── docs/
├── configs/
├── examples/
├── scripts/
├── tests/
└── src/
    └── project_name/
        ├── __init__.py
        ├── data/
        ├── preprocessing/
        ├── domain/
        ├── models/
        ├── inference/
        ├── orchestration/
        ├── config/
        └── utils/
```

### Layout rules

- Put reusable Python code under `src/project_name/`.
- Put examples in `examples/`, not mixed into core modules.
- Put operational scripts in `scripts/`, unless they are real CLI entrypoints.
- Put tests in `tests/` and mirror the main package structure where practical.
- Put configuration templates or defaults in `configs/` or in a clear `config/` package.
- Keep `utils/` small. If a utility group becomes domain-specific, promote it into a real package.

---

## Class design guidelines

### Preferred class types
Most classes should fall into one of these categories.

#### 1. Focused domain classes
These represent a single concept or behavior.

Examples:

- a scaler manager
- a fetch client
- a task head
- a parser
- a feature transformer

Rules:

- one clear responsibility
- small public API
- methods that reflect the lifecycle of the object
- minimal hidden state

#### 2. Configuration-backed component classes
Use dataclasses when a class mostly defines structured inputs plus one main action.

Examples:

- pipeline steps
- plugins
- block-style components
- inference stage definitions

Rules:

- prefer explicit fields
- validate in `__post_init__` when needed
- keep the main behavior method small and well named

#### 3. Orchestration classes
These coordinate multiple components.

Examples:

- trainer
- pipeline runner
- streaming coordinator
- workflow executor

Rules:

- should assemble and sequence work, not absorb every detail
- should delegate domain logic to focused helper classes
- should be the exception, not the default
- if the class keeps growing, split responsibilities early

### Class scope rules

A class is probably too large if it does several of the following:

- owns configuration parsing
- validates user input
- performs transformation logic
- manages external I/O
- logs reporting state
- handles retries or recovery
- formats output

If a class does more than two or three of these, split it.

### Inheritance rules

Use inheritance only when it improves clarity for extension.

Good uses:

- a tiny abstract base class for plugins
- a shared protocol for model/task interfaces
- a common interface for interchangeable pipeline components

Bad uses:

- inheriting only to reuse one helper method
- deep hierarchies where behavior is spread across multiple parent classes
- base classes with optional behavior toggled by many flags

---

## Function and method style

### Keep functions small and direct
A reader should usually understand a function in one pass.

Prefer:

- one job per function
- descriptive names
- explicit parameters
- return values over hidden mutation

Avoid:

- functions that parse config, transform data, write files, and report metrics all at once
- heavily nested control flow
- side effects that are not visible from the function name

### Suggested method naming
Use names that reveal intent.

Prefer:

- `load_config`
- `validate_inputs`
- `transform_features`
- `predict_batch`
- `write_results`
- `build_pipeline`

Avoid vague names like:

- `run`
- `handle`
- `process`
- `do_work`

These are acceptable only when the surrounding class makes the scope obvious.

---

## Comments and docstrings

### Comment style
Comments should be concise, human-readable, and useful.

Use comments to explain:

- why a choice was made
- non-obvious domain assumptions
- invariants
- edge cases
- external system behavior

Do not use comments to narrate obvious code.

Good:

```python
# Use UTC here because upstream timestamps may arrive in mixed local formats.
```

Bad:

```python
# Increment i by 1.
i += 1
```

### Docstrings
Public modules, classes, and important functions should have docstrings.

Preferred characteristics:

- explain purpose first
- mention key inputs and outputs
- describe important constraints
- keep them shorter than the code they describe

Good docstrings are especially important for:

- public APIs
- configuration objects
- extension points
- orchestration classes
- anything an agent is likely to reuse

### Readability rules for comments and docs

- keep comments short
- use complete thoughts, not fragments where possible
- avoid excessive jargon
- prefer direct language
- do not leave stale TODO-style comments without context

---

## Configuration guidelines

Configuration should be flexible, explicit, and validated.

### Preferred patterns

- central config schema or typed config objects
- documented defaults
- environment-specific overrides kept separate from code
- configuration loaded once near the boundary, then passed as structured objects

### Recommended approaches

Depending on repo size and complexity, use one of:

- dataclasses
- `TypedDict`
- pydantic models
- clearly validated dictionaries

### Configuration rules

- validate as early as possible
- reject unknown or malformed values where appropriate
- avoid mutating shared config in place
- keep naming consistent across config files and Python code
- separate user config from derived runtime state

### Example pattern

```python
from dataclasses import dataclass


@dataclass(slots=True)
class InferenceConfig:
    batch_size: int = 128
    device: str = "cpu"
    score_threshold: float = 0.5

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
```

---

## Error-handling guidelines

### General rules

- validate at the boundary
- raise specific exceptions
- include actionable messages
- log context where operations may fail in production
- only catch exceptions when you can add value

### Preferred exception types
Use built-in exceptions unless a custom exception adds real clarity.

Examples:

- `ValueError` for invalid parameter values
- `TypeError` for wrong types when not already enforced
- `KeyError` for missing required keys
- `FileNotFoundError` for missing files
- `RuntimeError` for invalid runtime state
- `ImportError` for missing optional dependencies

### Good error messages
Good messages should identify:

- what failed
- which input or state caused it
- what the caller can fix

Good:

```python
raise ValueError(f"Unknown model name: {model_name}. Expected one of {sorted(valid_models)}")
```

Bad:

```python
raise Exception("bad model")
```

### Catching exceptions
Catch exceptions only when you will:

- add context
- convert them into a better public error
- clean up resources
- implement justified retry or fallback behavior
- keep a long-running service alive

If catching broadly, explain why.

Good:

```python
try:
    client.connect()
except OSError as exc:
    raise RuntimeError(f"Failed to connect to metadata service at {host}") from exc
```

Broad catches such as `except Exception` should be limited to:

- top-level CLI boundaries
- service loops that must keep running
- cleanup/reporting layers

Even there, they should:

- log enough context
- avoid hiding the root cause
- document why broad handling is needed

### `assert` usage
Use `assert` only for internal invariants that should never fail in normal operation.

Do not use `assert` for:

- validating user config
- checking external input
- data file validation
- runtime service conditions

---

## Logging guidelines

Logging should support debugging and operations without drowning the user in noise.

### Log when

- entering or completing major workflow steps
- loading external resources
- retrying or recovering from operational issues
- catching and re-raising with context
- reporting summary outcomes

### Do not log

- every trivial internal step
- values already obvious from nearby code
- duplicate error messages at many layers

### Logging style

- prefer structured, descriptive messages
- include identifiers that help debugging
- keep logs concise
- do not use `print` in reusable library code

---

## Testing expectations

A new repo should follow `salt` more than the other two repos here.

### Minimum expectations

- unit tests for core domain logic
- tests for configuration validation
- tests for error cases
- at least one end-to-end or integration-style test for major workflows

### Good test targets

- small utility functions
- data transforms
- class validation behavior
- pipeline assembly
- CLI or entrypoint smoke tests

### Test design rules

- keep tests readable
- use descriptive test names
- test both success and failure cases
- avoid over-mocking when simple fixtures will do

---

## Tooling expectations

A new repo should define quality checks early.

### Recommended baseline

- `pyproject.toml` for project metadata and tool config
- Ruff for linting and formatting
- pytest for tests
- mypy or another type checker for important modules
- pre-commit for consistent local checks

### Style tooling philosophy
Use tooling to enforce consistency, but keep the rule set practical.

Tooling should help readability, not produce ceremony.

---

## Guidance for future agents

When adding new code to a repo that follows this guide:

### Before writing code

- identify the package boundary where the change belongs
- decide whether the new logic is a function, focused class, or orchestration change
- check whether configuration or validation should be added at the boundary
- prefer extending an existing clear pattern over inventing a new local style

### While writing code

- optimize for clarity first
- keep comments concise and meaningful
- make exceptions specific and actionable
- avoid growing one class into a catch-all coordinator
- preserve explicit control flow

### When introducing a class
Ask:

- does this class have one clear responsibility?
- does it own state that justifies being a class?
- can it be explained in one sentence?
- would a small function be simpler?

If the answer is no, reconsider the abstraction.

### When introducing config
Ask:

- is the configuration documented?
- are defaults explicit?
- is invalid input rejected early?
- is runtime-derived state separated from user input?

### When handling errors
Ask:

- is this the right exception type?
- will the message help a future maintainer?
- am I catching too broadly?
- am I hiding a root cause that should remain visible?

---

## Recommended defaults for a new repo

If no stronger project-specific constraint exists, use these defaults.

### Architecture
- responsibility-based package layout
- library code separate from scripts and examples
- composition-oriented design

### Classes
- small focused classes by default
- dataclasses for config-heavy components
- minimal base classes only for extension points
- orchestration classes kept thin

### Readability
- descriptive names
- short methods
- concise comments explaining why
- docstrings on public APIs and key extension points

### Configuration
- typed or validated configuration
- explicit defaults
- clear naming
- no hidden fallbacks

### Errors
- specific exceptions
- actionable messages
- broad catches only at justified boundaries
- logging with context, not noise

### Quality
- tests for core logic and failure paths
- linting and formatting in `pyproject.toml`
- type hints in important modules

---

## Final summary

The best combined style from these reference repos is:

- **architecture from `atom`**: clear package boundaries and visible workflow structure
- **engineering discipline from `salt`**: tests, type hints, docstrings, tooling, and validation
- **component design from `training-dataset-dumper`**: small, configuration-friendly, extension-oriented classes

Future repos should aim for code that is:

- easy to navigate
- easy to reason about
- easy to configure
- easy to extend
- easy to debug

When in doubt, prefer the design that a careful human can understand in one read.
