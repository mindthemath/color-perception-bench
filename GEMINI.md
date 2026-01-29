# Gemini Agent Instructions

This file outlines the architecture, style, and conventions for the **Color Perception Benchmark** project. Use this as a reference when modifying code or adding features.

## Project Overview
This project benchmarks multimodal embedding models by evaluating the alignment between text descriptions of colors (e.g., "royal blue") and their visual representation (solid color images).

### Core Components
- **CLI (`src/color_perception_bench/cli.py`)**: Interactive entry point using `questionary` and `rich`.
- **Benchmark Engine (`src/color_perception_bench/benchmark.py`)**: Orchestrates the async fetching of embeddings and calculation of metrics.
- **Providers (`src/color_perception_bench/providers/`)**:
  - `base.py`: Defines the `AsyncEmbeddingProvider` protocol and `BaseAsyncProvider` abstract base class.
  - `local.py` & `openai_compatible.py`: Concrete implementations.
- **Registry (`src/color_perception_bench/registry.py`)**: Manages `models.yaml`.
- **Cache (`src/color_perception_bench/cache.py`)**: Handles `joblib`-based caching of embeddings to minimize API costs/time.

## Tech Stack
- **Language**: Python 3.12+
- **Dependency Manager**: `uv` (uses `pyproject.toml` and `uv.lock`)
- **Async I/O**: `asyncio`, `aiohttp`
- **CLI/UI**: `rich` (output), `questionary` (input), `typer` (if applicable, currently seems manual)
- **Data**: `numpy`, `pandas` (implicit via results handling), `scikit-learn` (for cosine distance if needed, but currently manual calculation using numpy), `joblib` (caching).

## Architecture & patterns

### 1. Provider Pattern
New model providers must implement the `AsyncEmbeddingProvider` protocol (or inherit from `BaseAsyncProvider`).
- **Validation**: Providers *must* implement `validate_endpoints()` to check against OpenAPI schemas where possible.
- **Batching**: Implement `discover_batch_support()` to auto-detect optimal batch sizes.

### 2. Caching
Embeddings are cached per model in `cache/embeddings_<safe_model_name>.joblib`.
- **Always** check cache before fetching.
- **Invalidation**: Controlled via `force_refresh` flags.

### 3. Error Handling
- Use `ProviderValidationError` for configuration issues.
- Implement exponential backoff for rate limits (handled in `benchmark.py`'s `_fetch_with_retry`).

## Code Style & Conventions

### Formatting
- Follow **PEP 8**.
- Use **Type Hints** everywhere.
- **Docstrings**: Google style. Include `Args`, `Returns`, and `Raises`.

### CLI Output
- Use `rich.console.Console` for output.
- Use `rich.progress` for long-running tasks.
- **Do not** use standard `print()` for status updates; use `console.print()`.

### Async/Await
- The core loop is async. Ensure all I/O bound operations (API calls) are non-blocking.
- Use `asyncio.gather` for parallel processing of models.

## Common Tasks for Agents

### Adding a New Provider Type
1. Create `src/color_perception_bench/providers/your_provider.py`.
2. Inherit from `BaseAsyncProvider`.
3. Implement abstract methods (`_ensure_session`, `validate_endpoints`, `discover_batch_support`, `get_text_embeddings`, `get_image_embeddings`).
4. Register it in `src/color_perception_bench/registry.py` (if dynamic loading isn't implemented).

### Modifying Metrics
- Metrics calculation happens in `benchmark.py`.
- Results are appended to `benchmark_results.tsv`.
- If adding a new metric, update:
  1. The calculation logic in `benchmark.py`.
  2. The TSV header writing logic (ensure backward compatibility or handle schema changes).
  3. The `rich` table display.

## File Structure
- `models.yaml`: **Source of truth** for model configs.
- `.env`: **Source of truth** for secrets.
- `src/`: Application source code. Do not put scripts in root unless they are administrative (like `Makefile`).

## Workflow
1. **Analyze**: Understand the goal.
2. **Check**: Run `uv sync` if dependencies might be missing.
3. **Implement**: Modify code, respecting the async architecture.
4. **Verify**: Run `uv run color-perception-bench` to test the CLI flow.
