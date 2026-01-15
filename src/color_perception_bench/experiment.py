import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from color_perception_bench.colors import XKCD_COLORS_RGB, create_swatch_image
from color_perception_bench.embeddings import get_image_embedding, get_text_embedding

CACHE_FILE = "embeddings_cache.joblib"


def get_embeddings():
    """Fetch or load embeddings for all XKCD colors."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading embeddings from {CACHE_FILE}...")
        return joblib.load(CACHE_FILE)

    print("Fetching embeddings...")
    data = {}

    # Iterate over all colors
    for name, rgb in tqdm(XKCD_COLORS_RGB.items(), desc="Processing colors"):
        try:
            # Get text embedding for the color name
            text_emb = get_text_embedding(name)

            # Generate swatch and get image embedding
            swatch = create_swatch_image(rgb)
            img_emb = get_image_embedding(swatch)

            data[name] = {
                "rgb": rgb,
                "text_embedding": text_emb,
                "image_embedding": img_emb,
            }
        except Exception as e:
            print(f"Error processing {name}: {e}")

    print(f"Saving embeddings to {CACHE_FILE}...")
    joblib.dump(data, CACHE_FILE)
    return data


def compute_distances(data):
    """Compute pairwise distances for RGB and embeddings."""
    names = list(data.keys())
    n = len(names)

    if n < 2:
        print("Not enough data to calculate correlations.")
        return None

    # Prepare arrays
    # RGB: (N, 3)
    rgbs = np.array([data[name]["rgb"] for name in names])

    # Embeddings: (N, D)
    text_embs = np.array([data[name]["text_embedding"] for name in names])
    img_embs = np.array([data[name]["image_embedding"] for name in names])

    # Normalize embeddings to ensure cosine distance is just 1 - dot
    # Avoid division by zero if any zero vectors exist (unlikely but safe)
    text_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_norms = np.linalg.norm(img_embs, axis=1, keepdims=True)

    # Replace zero norms with 1 to avoid NaN (though embedding usually non-zero)
    text_norms[text_norms == 0] = 1
    img_norms[img_norms == 0] = 1

    text_embs = text_embs / text_norms
    img_embs = img_embs / img_norms

    # 1. Pairwise RGB Euclidean distances
    print("Computing RGB distances...")
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    rgb_diff = rgbs[:, np.newaxis, :] - rgbs[np.newaxis, :, :]
    rgb_dists = np.linalg.norm(rgb_diff, axis=2)

    # 2. Pairwise Image Embedding Cosine Distances
    # Cosine Dist = 1 - Similarity
    print("Computing Image Embedding distances...")
    img_sims = np.dot(img_embs, img_embs.T)
    img_dists = 1 - img_sims

    # 3. Pairwise Text Embedding Cosine Distances
    print("Computing Text Embedding distances...")
    text_sims = np.dot(text_embs, text_embs.T)
    text_dists = 1 - text_sims

    # 4. Same-color Text vs Image Embedding Distances
    print("Computing Cross-modal distances...")
    cross_modal_sims = np.sum(text_embs * img_embs, axis=1)
    cross_modal_dists = 1 - cross_modal_sims

    # Extract upper triangle indices (excluding diagonal)
    iu = np.triu_indices(n, k=1)

    rgb_flat = rgb_dists[iu]
    img_flat = img_dists[iu]
    text_flat = text_dists[iu]

    print(f"Analyzed {len(rgb_flat)} pairs.")

    return {
        "n": n,
        "rgb_flat": rgb_flat,
        "img_flat": img_flat,
        "text_flat": text_flat,
        "cross_modal_dists": cross_modal_dists,
    }


def plot_analysis(results, output_path="color_perception_correlation.png"):
    """Plot correlations and distributions from analysis results."""
    if not results:
        print("No results to plot.")
        return

    n = results["n"]
    rgb_flat = results["rgb_flat"]
    img_flat = results["img_flat"]
    text_flat = results["text_flat"]
    cross_modal_dists = results["cross_modal_dists"]

    print("Generating plots...")
    # Use GridSpec to create a layout with 2 rows: top row with 2 cols, bottom row with 1 col
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(
        "Color Perception Analysis: Spaces & Alignment",
        fontsize=32,
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])  # Spans both columns

    # Make top plots square
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    # Plot 1: RGB vs Image Embed
    # Using hexbin for better visualization of dense points
    hb1 = ax1.hexbin(
        rgb_flat, img_flat, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    ax1.set_xlabel("RGB Euclidean Distance", fontsize=20)
    ax1.set_ylabel("Image Embedding Cosine Distance", fontsize=20)
    ax1.set_title("Color Space vs Image Embeddings", fontsize=24)
    cb1 = fig.colorbar(hb1, ax=ax1)
    cb1.set_label("Log Count", fontsize=20)
    cb1.ax.tick_params(labelsize=16)

    # Plot 2: RGB vs Text Embed
    hb2 = ax2.hexbin(
        rgb_flat, text_flat, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    ax2.set_xlabel("RGB Euclidean Distance", fontsize=20)
    ax2.set_ylabel("Text Embedding Cosine Distance", fontsize=20)
    ax2.set_title("Color Space vs Text Embeddings", fontsize=24)
    cb2 = fig.colorbar(hb2, ax=ax2)
    cb2.set_label("Log Count", fontsize=20)
    cb2.ax.tick_params(labelsize=16)

    # Plot 3: Histogram of Text-Image Distances
    counts, bins, patches = ax3.hist(
        cross_modal_dists, bins=50, color="skyblue", edgecolor="black"
    )
    ax3.set_xlabel("Cosine Distance (Text vs Image)", fontsize=20)
    ax3.set_ylabel("Count", fontsize=20)
    ax3.set_title(f"Distribution of Text-Image Alignment (N={n})", fontsize=24)

    # Add stats
    median_dist = np.median(cross_modal_dists)
    mean_dist = np.mean(cross_modal_dists)

    # Vertical line
    ax3.axvline(
        median_dist,
        color="red",
        linestyle="dashed",
        linewidth=5,
    )

    # Add text annotation for median
    ax3.annotate(
        f"Median: {median_dist:.4f}",
        xy=(median_dist, 0.87),
        xycoords=ax3.get_xaxis_transform(),
        xytext=(0.75, 0.87),
        textcoords="axes fraction",
        color="red",
        fontsize=20,
        verticalalignment="center",
        arrowprops=dict(arrowstyle="->", color="red", linewidth=3),
    )

    # Vertical line
    ax3.axvline(
        mean_dist,
        color="blue",
        linestyle="dashed",
        linewidth=5,
    )

    # Add text annotation for mean
    ax3.annotate(
        f"Mean:    {mean_dist:.4f}",
        xy=(mean_dist, 0.92),
        xycoords=ax3.get_xaxis_transform(),
        xytext=(0.75, 0.92),
        textcoords="axes fraction",
        color="blue",
        fontsize=20,
        verticalalignment="center",
        arrowprops=dict(arrowstyle="->", color="blue", linewidth=3),
    )

    # Increase tick label sizes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis="both", which="major", labelsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    output_path = "color_perception_correlation.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    embeddings_data = get_embeddings()
    analysis_results = compute_distances(embeddings_data)
    plot_analysis(analysis_results)
