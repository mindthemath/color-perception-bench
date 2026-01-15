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


def compute_distances_and_plot(data):
    """Compute pairwise distances and plot correlations."""
    names = list(data.keys())
    n = len(names)

    if n < 2:
        print("Not enough data to calculate correlations.")
        return

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

    # Extract upper triangle indices (excluding diagonal)
    iu = np.triu_indices(n, k=1)

    rgb_flat = rgb_dists[iu]
    img_flat = img_dists[iu]
    text_flat = text_dists[iu]

    print(f"Analyzed {len(rgb_flat)} pairs.")

    # Plotting
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        "Pairwise Distance Correlation: RGB Color Space (Euclidean) vs Embedding Space (Cosine)",
        fontsize=16,
    )

    # Plot 1: RGB vs Image Embed
    # Using hexbin for better visualization of dense points
    hb1 = axes[0].hexbin(
        rgb_flat, img_flat, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    axes[0].set_xlabel("RGB Euclidean Distance")
    axes[0].set_ylabel("Image Embedding Cosine Distance")
    axes[0].set_title("Color Space vs Image Embeddings")
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label("Log Count")

    # Plot 2: RGB vs Text Embed
    hb2 = axes[1].hexbin(
        rgb_flat, text_flat, gridsize=50, cmap="plasma", mincnt=1, bins="log"
    )
    axes[1].set_xlabel("RGB Euclidean Distance")
    axes[1].set_ylabel("Text Embedding Cosine Distance")
    axes[1].set_title("Color Space vs Text Embeddings")
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label("Log Count")

    plt.tight_layout()
    output_path = "color_perception_correlation.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    # plt.show()


if __name__ == "__main__":
    embeddings_data = get_embeddings()
    compute_distances_and_plot(embeddings_data)
