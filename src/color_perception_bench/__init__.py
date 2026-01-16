"""Color Perception Benchmark - Multi-model evaluation of color-text alignment."""

from color_perception_bench.benchmark import (
    compute_alignment_metrics,
    fetch_embeddings_for_model,
    plot_model_analysis,
    run_benchmark,
)
from color_perception_bench.cache import (
    invalidate_cache,
    load_embeddings,
    save_embeddings,
)
from color_perception_bench.cli import main
from color_perception_bench.registry import (
    add_model,
    get_provider,
    list_models,
    remove_model,
)

__version__ = "0.2.0"

__all__ = [
    # Benchmark
    "run_benchmark",
    "fetch_embeddings_for_model",
    "compute_alignment_metrics",
    "plot_model_analysis",
    # Registry
    "add_model",
    "remove_model",
    "list_models",
    "get_provider",
    # Cache
    "load_embeddings",
    "save_embeddings",
    "invalidate_cache",
    # CLI
    "main",
]
