"""Async benchmark runner for multi-model color perception evaluation."""

import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from color_perception_bench.cache import load_embeddings, save_embeddings
from color_perception_bench.colors import XKCD_COLORS_RGB, create_swatch_image
from color_perception_bench.providers.base import ProviderValidationError
from color_perception_bench.registry import get_provider

if TYPE_CHECKING:
    from color_perception_bench.providers.local import LocalProvider
    from color_perception_bench.providers.openai_compatible import (
        OpenAICompatibleProvider,
    )

RESULTS_FILE = Path("benchmark_results.tsv")
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5.0  # seconds

console = Console()


async def _fetch_with_retry(fetch_func, batch, modality: str, max_retries=MAX_RETRIES):
    """
    Fetch embeddings with exponential backoff retry on rate limits.

    Args:
        fetch_func: The async function to call (get_text_embeddings or get_image_embeddings)
        batch: The batch of items to process
        modality: Either "text" or "image" for error messages
        max_retries: Maximum number of retry attempts

    Returns:
        List of embeddings

    Raises:
        RuntimeError: If all retries are exhausted
    """
    delay = INITIAL_RETRY_DELAY
    last_error = None

    for attempt in range(max_retries):
        try:
            return await fetch_func(batch)
        except RuntimeError as e:
            error_str = str(e)
            # Check if it's a rate limit error (429) or a gateway timeout (524)
            if (
                "429" in error_str
                or "rate limit" in error_str.lower()
                or "524" in error_str
                or "timeout" in error_str.lower()
            ):
                last_error = e
                if attempt < max_retries - 1:
                    console.print(
                        f"  [yellow]⚠[/yellow] Rate limit hit, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    console.print(
                        f"  [red]✗[/red] Rate limit exceeded after {max_retries} attempts"
                    )
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise

    # If we get here, all retries failed
    raise RuntimeError(
        f"{modality.capitalize()} embedding failed after {max_retries} retries"
    ) from last_error


async def fetch_embeddings_for_model(
    model_name: str,
    provider: "LocalProvider | OpenAICompatibleProvider",
    force_refresh: bool = False,
) -> dict:
    """
    Fetch or load embeddings for all XKCD colors using a specific model.

    Args:
        model_name: Name of the model.
        provider: Provider instance to use.
        force_refresh: If True, ignore cache and re-fetch.

    Returns:
        Dictionary mapping color names to their data (rgb, text_embedding, image_embedding).
    """
    # Check cache first
    if not force_refresh:
        cached = load_embeddings(model_name)
        if cached is not None:
            # Check if cache is complete (has both text and image embeddings)
            sample_entry = next(iter(cached.values()))
            if (
                sample_entry.get("image_embedding") is not None
                and sample_entry.get("text_embedding") is not None
            ):
                console.print(
                    f"[green]✓[/green] Loaded {len(cached)} colors from cache for [bold]{model_name}[/bold]"
                )
                return cached
            else:
                console.print(
                    f"[yellow]![/yellow] Found partial cache for [bold]{model_name}[/bold], completing..."
                )

    console.print(
        f"[blue]→[/blue] Fetching embeddings for [bold]{model_name}[/bold]..."
    )

    async with provider:
        # Validate endpoints
        try:
            await provider.validate_endpoints()
            console.print("  [green]✓[/green] Endpoints validated")
        except ProviderValidationError as e:
            console.print(f"  [red]✗[/red] Validation failed: {e}")
            raise

        # Discover batch support
        batch_config = await provider.discover_batch_support()
        effective_batch = provider.config.effective_batch_size
        if batch_config.supported:
            console.print(
                f"  [green]✓[/green] Batch support: max={batch_config.max_size}, "
                f"using={effective_batch}"
            )
        else:
            console.print(
                "  [yellow]![/yellow] No batch support, using sequential requests"
            )

        # Prepare data
        color_names = list(XKCD_COLORS_RGB.keys())
        color_rgbs = list(XKCD_COLORS_RGB.values())
        n_colors = len(color_names)

        # Check for partial cache
        cached_partial = None
        if not force_refresh:
            cached_partial = load_embeddings(model_name)
            if (
                cached_partial
                and cached_partial.get(color_names[0], {}).get("image_embedding")
                is not None
            ):
                cached_partial = None  # Cache is complete, already returned earlier

        data = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Fetch text embeddings (or use cached)
            text_embeddings = []
            if cached_partial:
                console.print("  [green]✓[/green] Using cached text embeddings")
                text_embeddings = [
                    cached_partial[name]["text_embedding"] for name in color_names
                ]
            else:
                text_task = progress.add_task(
                    f"  Text embeddings ({model_name})", total=n_colors
                )

                for i in range(0, n_colors, effective_batch):
                    batch_names = color_names[i : i + effective_batch]
                    batch_embs = await _fetch_with_retry(
                        provider.get_text_embeddings, batch_names, "text"
                    )
                    text_embeddings.extend(batch_embs)
                    progress.update(text_task, advance=len(batch_names))

                # Save partial cache with text embeddings only
                console.print(
                    "  [blue]→[/blue] Caching text embeddings (partial save)..."
                )
                partial_data = {}
                for i, name in enumerate(color_names):
                    partial_data[name] = {
                        "rgb": color_rgbs[i],
                        "text_embedding": text_embeddings[i],
                        "image_embedding": None,  # Placeholder
                    }
                save_embeddings(model_name, partial_data)

            # Fetch image embeddings in batches
            img_task = progress.add_task(
                f"  Image embeddings ({model_name})", total=n_colors
            )

            image_embeddings = []
            for i in range(0, n_colors, effective_batch):
                batch_rgbs = color_rgbs[i : i + effective_batch]
                batch_images = [create_swatch_image(rgb) for rgb in batch_rgbs]
                batch_embs = await _fetch_with_retry(
                    provider.get_image_embeddings, batch_images, "image"
                )
                image_embeddings.extend(batch_embs)
                progress.update(img_task, advance=len(batch_rgbs))

        # Assemble data
        for i, name in enumerate(color_names):
            data[name] = {
                "rgb": color_rgbs[i],
                "text_embedding": text_embeddings[i],
                "image_embedding": image_embeddings[i],
            }

        # Save to cache
        save_embeddings(model_name, data)
        console.print(f"  [green]✓[/green] Cached {len(data)} colors")

    return data


def compute_alignment_metrics(data: dict) -> dict:
    """
    Compute cross-modal alignment metrics from embeddings data.

    Args:
        data: Dictionary mapping color names to their embeddings.

    Returns:
        Dictionary with alignment metrics (mean, median, std, variance, n).
    """
    # Filter out metadata keys (like '_model_name')
    names = [k for k in data.keys() if not k.startswith("_")]
    n = len(names)

    if n < 1:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "variance": float("nan"),
        }

    # Extract embeddings
    text_embs = np.array([data[name]["text_embedding"] for name in names])
    img_embs = np.array([data[name]["image_embedding"] for name in names])

    # Normalize embeddings
    text_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_norms = np.linalg.norm(img_embs, axis=1, keepdims=True)

    text_norms[text_norms == 0] = 1
    img_norms[img_norms == 0] = 1

    text_embs = text_embs / text_norms
    img_embs = img_embs / img_norms

    # Compute cross-modal cosine distances (text vs image for same color)
    cross_modal_sims = np.sum(text_embs * img_embs, axis=1)
    cross_modal_dists = 1 - cross_modal_sims

    return {
        "n": n,
        "mean": float(np.mean(cross_modal_dists)),
        "median": float(np.median(cross_modal_dists)),
        "std": float(np.std(cross_modal_dists)),
        "variance": float(np.var(cross_modal_dists)),
        "min": float(np.min(cross_modal_dists)),
        "max": float(np.max(cross_modal_dists)),
        "distances": cross_modal_dists,  # For histogram if needed
    }


def plot_model_analysis(
    model_name: str, data: dict, output_dir: str | Path = "figs"
) -> Path:
    """
    Generate correlation plots for a specific model's embeddings.

    Args:
        model_name: Name of the model
        data: Dictionary mapping color names to embeddings
        output_dir: Directory to save plots (default: "figs")

    Returns:
        Path to the saved plot file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Use original model name from cache data if available (preserves periods)
    original_name = data.get("_model_name", model_name)

    # Sanitize model name for filename
    safe_name = original_name.replace("/", "_").replace(":", "_").replace(".", "_")
    output_path = output_dir / f"{safe_name}_correlation.png"

    # Filter out metadata keys
    names = [k for k in data.keys() if not k.startswith("_")]
    n = len(names)

    if n < 2:
        console.print(f"[red]Not enough data to plot for {original_name}[/red]")
        return output_path

    # Prepare arrays
    rgbs = np.array([data[name]["rgb"] for name in names])
    text_embs = np.array([data[name]["text_embedding"] for name in names])
    img_embs = np.array([data[name]["image_embedding"] for name in names])

    # Normalize embeddings
    text_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_norms = np.linalg.norm(img_embs, axis=1, keepdims=True)
    text_norms[text_norms == 0] = 1
    img_norms[img_norms == 0] = 1
    text_embs = text_embs / text_norms
    img_embs = img_embs / img_norms

    # Compute pairwise RGB distances and normalize to [0, 1]
    rgb_diff = rgbs[:, np.newaxis, :] - rgbs[np.newaxis, :, :]
    rgb_dists = np.linalg.norm(rgb_diff, axis=2)
    # Max distance for RGB values in [0, 255] is sqrt(3 * 255^2)
    rgb_dists = rgb_dists / (255.0 * np.sqrt(3))

    # Compute pairwise embedding distances
    img_sims = np.dot(img_embs, img_embs.T)
    img_dists = 1 - img_sims
    text_sims = np.dot(text_embs, text_embs.T)
    text_dists = 1 - text_sims

    # Cross-modal distances
    cross_modal_sims = np.sum(text_embs * img_embs, axis=1)
    cross_modal_dists = 1 - cross_modal_sims

    # Extract upper triangle (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    rgb_flat = rgb_dists[iu]
    img_flat = img_dists[iu]
    text_flat = text_dists[iu]

    # Create figure
    console.print(f"  Generating plots for [bold]{original_name}[/bold]...")
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(
        f"Color Perception Analysis: {original_name}",
        fontsize=32,
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    # Plot 1: RGB vs Image Embed
    hb1 = ax1.hexbin(
        rgb_flat, img_flat, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    ax1.set_xlabel("Normalized RGB Euclidean Distance", fontsize=20)
    ax1.set_ylabel("Image Embedding Cosine Distance", fontsize=20)
    ax1.set_title("Color Space vs Image Embeddings", fontsize=24)
    cb1 = fig.colorbar(hb1, ax=ax1)
    cb1.set_label("Log Count", fontsize=20)
    cb1.ax.tick_params(labelsize=16)

    # Plot 2: RGB vs Text Embed
    hb2 = ax2.hexbin(
        rgb_flat, text_flat, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    ax2.set_xlabel("Normalized RGB Euclidean Distance", fontsize=20)
    ax2.set_ylabel("Text Embedding Cosine Distance", fontsize=20)
    ax2.set_title("Color Space vs Text Embeddings", fontsize=24)
    cb2 = fig.colorbar(hb2, ax=ax2)
    cb2.set_label("Log Count", fontsize=20)
    cb2.ax.tick_params(labelsize=16)

    # Plot 3: Histogram of Text-Image Distances
    ax3.hist(cross_modal_dists, bins=50, color="skyblue", edgecolor="black")
    ax3.set_xlabel("Cosine Distance (Text vs Image)", fontsize=20)
    ax3.set_ylabel("Count", fontsize=20)
    ax3.set_title(f"Distribution of Text-Image Alignment (N={n})", fontsize=24)

    # Add stats
    median_dist = np.median(cross_modal_dists)
    mean_dist = np.mean(cross_modal_dists)

    ax3.axvline(median_dist, color="red", linestyle="dashed", linewidth=5)
    ax3.annotate(
        f"Median: {median_dist:.4f}",
        xy=(median_dist, 0.87),
        xycoords=ax3.get_xaxis_transform(),
        xytext=(0.05, 0.87),
        textcoords="axes fraction",
        color="red",
        fontsize=20,
        verticalalignment="center",
        arrowprops=dict(arrowstyle="->", color="red", linewidth=3),
    )

    ax3.axvline(mean_dist, color="blue", linestyle="dashed", linewidth=5)
    ax3.annotate(
        f"Mean:    {mean_dist:.4f}",
        xy=(mean_dist, 0.92),
        xycoords=ax3.get_xaxis_transform(),
        xytext=(0.05, 0.92),
        textcoords="axes fraction",
        color="blue",
        fontsize=20,
        verticalalignment="center",
        arrowprops=dict(arrowstyle="->", color="blue", linewidth=3),
    )

    # Set consistent axis limits for comparison
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_xlim(0, 1)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis="both", which="major", labelsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    console.print(f"  [green]✓[/green] Saved plot to [cyan]{output_path}[/cyan]")
    return output_path


def save_results_to_tsv(
    model_name: str, metrics: dict, timestamp: datetime | None = None
) -> None:
    """
    Append benchmark results to TSV file.

    Args:
        model_name: Name of the model.
        metrics: Dictionary with computed metrics.
        timestamp: Optional timestamp (defaults to now).
    """
    if timestamp is None:
        timestamp = datetime.now()

    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Write header if file is new
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "model_name",
                    "n_colors",
                    "mean_distance",
                    "median_distance",
                    "std_distance",
                    "variance",
                    "min_distance",
                    "max_distance",
                ]
            )

        writer.writerow(
            [
                timestamp.isoformat(),
                model_name,
                metrics["n"],
                f"{metrics['mean']:.6f}",
                f"{metrics['median']:.6f}",
                f"{metrics['std']:.6f}",
                f"{metrics['variance']:.6f}",
                f"{metrics['min']:.6f}",
                f"{metrics['max']:.6f}",
            ]
        )


def load_results_from_tsv() -> list[dict]:
    """
    Load all results from TSV file.

    Returns:
        List of result dictionaries.
    """
    if not RESULTS_FILE.exists():
        return []

    results = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(row)

    return results


def print_results_table(results: list[dict] | None = None) -> None:
    """
    Print results in a nice table format.

    Args:
        results: List of result dicts (loads from file if None).
    """
    if results is None:
        results = load_results_from_tsv()

    if not results:
        console.print("[yellow]No benchmark results found.[/yellow]")
        return

    table = Table(title="Color Perception Benchmark Results")

    table.add_column("Timestamp", style="dim")
    table.add_column("Model", style="cyan bold")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Median", justify="right", style="green")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="red")

    for row in results:
        # Format timestamp to be shorter
        ts = row.get("timestamp", "")
        if "T" in ts:
            ts = ts.split("T")[0] + " " + ts.split("T")[1][:8]

        table.add_row(
            ts,
            row.get("model_name", ""),
            row.get("n_colors", ""),
            row.get("mean_distance", ""),
            row.get("median_distance", ""),
            row.get("std_distance", ""),
            row.get("min_distance", ""),
            row.get("max_distance", ""),
        )

    console.print(table)


async def run_benchmark(
    model_names: list[str], force_refresh: bool = False
) -> dict[str, dict]:
    """
    Run benchmark for multiple models.

    Args:
        model_names: List of model names to benchmark.
        force_refresh: If True, ignore cache.

    Returns:
        Dictionary mapping model names to their metrics.
    """
    all_metrics = {}
    timestamp = datetime.now()

    for model_name in model_names:
        console.print()
        console.rule(f"[bold blue]{model_name}[/bold blue]")

        try:
            provider = get_provider(model_name)
            data = await fetch_embeddings_for_model(
                model_name, provider, force_refresh=force_refresh
            )

            metrics = compute_alignment_metrics(data)
            all_metrics[model_name] = metrics

            # Save to TSV
            save_results_to_tsv(model_name, metrics, timestamp)

            # Print summary
            console.print(
                f"  [bold]Results:[/bold] mean={metrics['mean']:.4f}, "
                f"median={metrics['median']:.4f}, std={metrics['std']:.4f}"
            )

        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            all_metrics[model_name] = {"error": str(e)}

    # Print final summary table
    console.print()
    console.rule("[bold green]Benchmark Complete[/bold green]")

    # Show just the new results
    new_results = []
    for model_name, metrics in all_metrics.items():
        if "error" not in metrics:
            new_results.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "model_name": model_name,
                    "n_colors": str(metrics["n"]),
                    "mean_distance": f"{metrics['mean']:.6f}",
                    "median_distance": f"{metrics['median']:.6f}",
                    "std_distance": f"{metrics['std']:.6f}",
                    "min_distance": f"{metrics['min']:.6f}",
                    "max_distance": f"{metrics['max']:.6f}",
                }
            )

    if new_results:
        print_results_table(new_results)

    return all_metrics
