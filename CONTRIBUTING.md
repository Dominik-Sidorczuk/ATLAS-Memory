# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS! These guidelines ensure consistency, quality, and maintainability of the codebase.

## Process

1. **Fork + branch** — work on your own branch: `git checkout -b feat/your-change-description`.
2. **Small, atomic commits** — each change (bugfix, feature, docs) is a separate commit with a descriptive message.
3. **Zero-trust verification** — before opening a PR, verify locally:
   ```bash
   pixi run test        # test suite
   pixi run doctor      # Repository Health Score ≥ 90/100
   pixi run ruff check .
   ```
4. **Doc-Sync** — every code change requires updating `README.md` and `ARCHITECTURE.md` in the same commit (do not leave documentation behind code).
5. **PR** — open a Pull Request with a description: what it changes, why, how to verify.

## Code Standards

- Python 3.11+, `from __future__ import annotations`
- 100% of data models in **Pydantic v2** (`Field` with `ge`/`le`/`default`)
- **No `pass` statements** in production code — raise `NotImplementedError` or log the event instead
- **No `torch` in hot path** — runtime layer (atlas_provider) must not import `torch`
- `sys.path.append`, **never** `insert(0)`
- Docstrings in English (project convention for public-facing code)

## Tests

- New features require unit tests (`tests/unit/`)
- Benchmarks require tests in `tests/benchmark/` (prefix `test_` is mandatory — files `benchmark_*.py` are not collected by pytest)
- Mutational tests (intentional error injection) are required for critical algorithms

## Reports with Numbers

Every report containing quantitative data (latencies, token usage, compression) **must** have coverage in raw JSON files in `docs/baselines/`. Reports without source data are treated as architectural targets, not empirical facts.

## License

This project is distributed under the [MIT License](LICENSE). By submitting a PR, you agree to make your contribution available under the same terms.
