# Multi-Model Color Perception Benchmark

A benchmark tool for evaluating how well multimodal embedding models align text and visual representations of colors.

## Features

- 🔄 **Async Processing**: Efficient parallel fetching of embeddings with automatic batching
- 🔍 **OpenAPI Validation**: Automatically validates endpoints against `/openapi.json` schema
- 📦 **Smart Batching**: Auto-discovers batch support and optimal batch sizes (4, 8, 16, 32, 64, 128, 256, 512, 1024)
- 💾 **Intelligent Caching**: Per-model caching to avoid redundant API calls
- 📊 **TSV Results**: Persistent results tracking with timestamp, mean/median/std metrics
- 🎨 **Interactive CLI**: Menu-driven interface with questionary

## Installation & Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys (e.g., OPENAI_API_KEY)
   ```

## Quick Start

### Launch the CLI

```bash
uv run color-perception-bench
```

### Common Tasks

#### 1. Add Models
The `local-default` model is auto-created on first run (pointing to `http://localhost:8080`).

To add an OpenAI-compatible model:
1. Select **Manage Models** → **Add Model**
2. **Name**: `openai-text-3-large` (example)
3. **Provider type**: `openai_compatible`
4. **Base URL**: `https://api.openai.com`
5. **Endpoints**: `/v1/embeddings` (for both text and image usually, or specific ones)
6. **API Key Env Var**: `OPENAI_API_KEY`

#### 2. Run Benchmark
1. Select **Run Benchmark**
2. Select models using Space, confirm with Enter.
3. Choose whether to force refresh the cache.
4. Watch the progress bars.

#### 3. View Results
- Select **View Last Results** in the CLI.
- Or view the raw file:
  ```bash
  cat benchmark_results.tsv
  ```

## Configuration

### Environment Variables
Edit `.env` to store your secrets:
```bash
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
```

### Model Registry
Models are stored in `models.yaml` (git-tracked).

### Provider Types
- **`local`**: For custom APIs or localhost servers with OpenAPI specs.
- **`openai_compatible`**: For OpenAI, Together AI, Anyscale, Fireworks, Replicate, etc.

## Architecture

```
src/color_perception_bench/
├── providers/
│   ├── base.py                  # AsyncEmbeddingProvider protocol
│   ├── local.py                 # Local API provider
│   └── openai_compatible.py     # OpenAI-style API provider
├── benchmark.py                 # Async benchmark runner
├── cache.py                     # Per-model caching layer
├── cli.py                       # Interactive menu interface
├── registry.py                  # Model configuration management
├── colors.py                    # XKCD color data (949 colors)
└── experiment.py                # Original POC (legacy)
```

## Metrics Explained

The benchmark computes **cross-modal alignment** between text and image embeddings for the same color:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mean** | Average cosine distance | Overall alignment quality |
| **Median** | Middle value | Robust to outliers |
| **Std** | Standard deviation | Consistency of alignment |
| **Min** | Best alignment | Best case performance |
| **Max** | Worst alignment | Worst case performance |

**Lower distances = Better alignment** between text and image embeddings.

## Python API Usage

You can also use the library programmatically:

```python
import asyncio
from color_perception_bench import (
    run_benchmark,
    add_model,
    list_models,
    print_results_table,
)

# 1. Add a model programmatically
add_model(
    name="openai-text-3-large",
    provider_type="openai_compatible",
    base_url="https://api.openai.com",
    text_endpoint="/v1/embeddings",
    image_endpoint="/v1/embeddings",
    api_key_env_var="OPENAI_API_KEY",
    batch_size=128
)

# 2. Run benchmark
asyncio.run(run_benchmark(["local-default", "openai-text-3-large"]))

# 3. Print results
print_results_table()
```

## Troubleshooting

- **"Import could not be resolved"**: Run `uv sync` and ensure `.venv` is activated.
- **"API key environment variable not set"**: Check `.env` and ensure the variable name matches the config.
- **"Failed to fetch OpenAPI schema"**: Ensure the provider is running and the URL is correct (must serve `/openapi.json`).
- **Cache not being used**: Check `cache/` directory. Use `force_refresh=True` to rebuild.
