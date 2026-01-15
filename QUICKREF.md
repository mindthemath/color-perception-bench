# Quick Reference Card

## Installation & Setup

```bash
# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Edit .env and add API keys
# vim .env
```

## Launch CLI

```bash
uv run color-perception-bench
```

## Common Tasks

### Add Your Local Model (Already Done)
The `local-default` model is auto-created on first run, pointing to `http://localhost:8080`.

### Add an OpenAI-Compatible Model

Via CLI:
1. `Manage Models` → `Add Model`
2. Name: `openai-text-3-large`
3. Provider type: `openai_compatible`
4. Base URL: `https://api.openai.com`
5. Text endpoint: `/v1/embeddings`
6. Image endpoint: `/v1/embeddings`
7. API key var: `OPENAI_API_KEY`

Via Python:
```python
from color_perception_bench.registry import add_model

add_model(
    name="openai-text-3-large",
    provider_type="openai_compatible",
    base_url="https://api.openai.com",
    text_endpoint="/v1/embeddings",
    image_endpoint="/v1/embeddings",
    api_key_env_var="OPENAI_API_KEY",
    batch_size=128  # Optional
)
```

### Run Benchmark

Via CLI:
1. `Run Benchmark`
2. Select models (space to select, enter to confirm)
3. Force refresh? (y/n)
4. Watch progress

Via Python:
```python
import asyncio
from color_perception_bench import run_benchmark

asyncio.run(run_benchmark(["local-default", "openai-text-3-large"]))
```

### View Results

Via CLI:
- `View Last Results`

Via Python:
```python
from color_perception_bench.benchmark import print_results_table
print_results_table()
```

Via file:
```bash
cat benchmark_results.tsv
# Or open in Excel/Google Sheets
```

### Clear Cache

Via CLI:
1. `Clear Cache`
2. Select model or "Clear ALL"
3. Confirm

Via Python:
```python
from color_perception_bench.cache import invalidate_cache
invalidate_cache("local-default")
```

## File Locations

```
.
├── .env                     # Your API keys (git-ignored)
├── models.yaml              # Model registry (git-tracked)
├── benchmark_results.tsv    # Results log (append-only)
└── cache/                   # Per-model caches (git-ignored)
    ├── embeddings_local_default.joblib
    └── embeddings_*.joblib
```

## Provider Types

### `local`
For custom APIs with OpenAPI spec:
- Localhost servers
- Self-hosted embedding services
- Custom implementations

### `openai_compatible`
For OpenAI-style APIs:
- OpenAI
- Together AI
- Anyscale
- Fireworks
- Modal
- Replicate
- etc.

## Batch Sizes

Valid options: `1, 4, 8, 16, 32, 64, 128, 256, 512, 1024`

- `auto` (default): Discovers from OpenAPI schema
- Manual: Override in model config

## Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mean** | Average cosine distance | Overall alignment quality |
| **Median** | Middle value | Robust to outliers |
| **Std** | Standard deviation | Consistency of alignment |
| **Min** | Best alignment | Best case performance |
| **Max** | Worst alignment | Worst case performance |

**Lower distances = Better alignment** between text and image embeddings for the same color.

## Troubleshooting

### "Import could not be resolved"
- Run: `uv sync`
- Check: `.venv` is activated

### "API key environment variable not set"
- Check: `.env` file exists
- Check: Variable name matches config
- Check: No spaces around `=` in `.env`

### "Failed to fetch OpenAPI schema"
- Check: Provider is running
- Check: URL is correct
- Check: `/openapi.json` endpoint exists

### "Text endpoint not found in OpenAPI schema"
- Check: Endpoint path is correct
- Check: OpenAPI spec is up to date
- Try: List available paths in error message

### Cache not being used
- Check: `cache/` directory exists
- Check: Model name matches exactly
- Force refresh: Use `force_refresh=True` flag

## Performance Tips

1. **Use batching**: Larger batch sizes = fewer API calls
2. **Use cache**: Run once, analyze many times
3. **Test subset first**: Use a small model for testing
4. **Monitor progress**: Rich progress bars show ETA

## Example Workflow

```bash
# 1. Setup
cp .env.example .env
vim .env  # Add API keys

# 2. Launch CLI
uv run color-perception-bench

# 3. Add models (if needed)
# Manage Models > Add Model

# 4. Run benchmark
# Run Benchmark > Select models > Run

# 5. View results
# View Last Results

# 6. Analyze
cat benchmark_results.tsv
# Open in spreadsheet for charts
```

## Python API

```python
import asyncio
from color_perception_bench import (
    run_benchmark,
    add_model,
    list_models,
    get_provider,
    print_results_table,
)

# List registered models
models = list_models()
print(models)

# Run benchmark
results = asyncio.run(run_benchmark(models))

# Print table
print_results_table()

# Access provider directly
provider = get_provider("local-default")
async with provider:
    await provider.validate_endpoints()
    # Use provider...
```

## Support

- Documentation: [USAGE.md](USAGE.md)
- Implementation: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- Original POC: [experiment.py](src/color_perception_bench/experiment.py)
