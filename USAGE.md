# Multi-Model Color Perception Benchmark

A benchmark tool for evaluating how well multimodal embedding models align text and visual representations of colors.

## Features

- 🔄 **Async Processing**: Efficient parallel fetching of embeddings with automatic batching
- 🔍 **OpenAPI Validation**: Automatically validates endpoints against `/openapi.json` schema
- 📦 **Smart Batching**: Auto-discovers batch support and optimal batch sizes (4, 8, 16, 32, 64, 128, 256, 512, 1024)
- 💾 **Intelligent Caching**: Per-model caching to avoid redundant API calls
- 📊 **TSV Results**: Persistent results tracking with timestamp, mean/median/std metrics
- 🎨 **Interactive CLI**: Menu-driven interface with questionary

## Installation

```bash
uv sync
```

## Quick Start

### Launch the Interactive CLI

```bash
uv run color-perception-bench
```

Or if installed:

```bash
color-perception-bench
```

### CLI Menu Options

1. **🚀 Run Benchmark** - Execute benchmark on selected models
2. **📋 View Last Results** - Display results table from TSV
3. **⚙️ Manage Models** - Add/remove/edit model configurations
   - ➕ Add Model
   - 📝 List Models
   - 🔧 Edit Batch Size
   - 🗑️ Remove Model
4. **🗑️ Clear Cache** - Invalidate cached embeddings for specific models
5. **❌ Exit**

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# OpenAI API key
OPENAI_API_KEY=sk-...

# Together AI API key
TOGETHER_API_KEY=...

# Add other providers as needed
```

### Model Registry

Models are stored in `models.yaml`. The CLI automatically creates a `local-default` model pointing to `http://localhost:8080`.

#### Add a Model

Via CLI:
- Select "Manage Models" > "Add Model"
- Follow the prompts

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
    batch_size=128  # Optional: override auto-detected batch size
)
```

#### Supported Provider Types

- `local` - For localhost or custom APIs with OpenAPI spec
- `openai_compatible` - For OpenAI, Together, Anyscale, Fireworks, etc.

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

## Metrics

The benchmark computes **cross-modal alignment** between text and image embeddings for the same color:

- **Mean Distance**: Average cosine distance between text and image embeddings
- **Median Distance**: Median cosine distance (robust to outliers)
- **Std/Variance**: Distribution spread
- **Min/Max**: Range of alignment quality

Lower distances indicate better text-image alignment.

## Output

### Console Output

Rich-formatted tables and progress bars with:
- Live batch progress per model
- Endpoint validation status
- Batch configuration confirmation
- Summary statistics

### TSV File

Results are appended to `benchmark_results.tsv`:

```tsv
timestamp	model_name	n_colors	mean_distance	median_distance	std_distance	variance	min_distance	max_distance
2026-01-15T...	local-default	949	0.234567	0.228901	0.045678	0.002087	0.123456	0.456789
```

### Cache Directory

Per-model cache files stored in `cache/`:

```
cache/
├── embeddings_local_default.joblib
├── embeddings_openai_text_3_large.joblib
└── ...
```

## API Requirements

Your embedding API must:

1. Provide `/openapi.json` endpoint
2. Support POST requests with JSON payloads
3. Accept text strings or base64-encoded images
4. Return embedding vectors

### Local API Example

If running a local embedding server:

```bash
# Start your embedding service on localhost:8080
# Ensure it serves /openapi.json, /txt/embed, /img/embed
```

### OpenAI-Compatible Example

For OpenAI or compatible providers:

```python
add_model(
    name="openai-text-3-large",
    provider_type="openai_compatible",
    base_url="https://api.openai.com",
    text_endpoint="/v1/embeddings",
    image_endpoint="/v1/embeddings",
    api_key_env_var="OPENAI_API_KEY"
)
```

## Programmatic Usage

```python
import asyncio
from color_perception_bench import run_benchmark, print_results_table

# Run benchmark for specific models
results = asyncio.run(run_benchmark(["local-default", "openai-text-3-large"]))

# View results
print_results_table()
```

## Development

### Project Structure

- `models.yaml` - Model registry (git tracked)
- `.env` - API keys (git ignored)
- `cache/` - Embedding cache (git ignored)
- `benchmark_results.tsv` - Results log (optional tracking)

### Adding New Provider Types

Extend `BaseAsyncProvider` in a new file under `providers/`:

```python
from color_perception_bench.providers.base import BaseAsyncProvider

class CustomProvider(BaseAsyncProvider):
    async def validate_endpoints(self) -> None:
        # Implement validation
        ...
```

## License

See LICENSE file for details.
