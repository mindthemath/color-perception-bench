import base64
import io

import requests
from PIL import Image

TEXT_EMBEDDING_ENDPOINT = "http://localhost:8080/txt/embed"
IMAGE_EMBEDDING_ENDPOINT = "http://localhost:8080/img/embed"


def get_text_embedding(text: str) -> list[float]:
    """Get text embedding from the custom API."""
    response = requests.post(TEXT_EMBEDDING_ENDPOINT, json={"input": text})
    response.raise_for_status()
    return response.json()["embedding"]


def get_image_embedding(image: Image.Image) -> list[float]:
    """Get image embedding from the custom API using a PIL Image."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    content = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = requests.post(IMAGE_EMBEDDING_ENDPOINT, json={"content": content})
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_distance(vec, other_vec):
    """Compute the cosine distance between two vectors."""
    from numpy import dot
    from numpy.linalg import norm

    return 1 - dot(vec, other_vec)  # / (norm(vec) * norm(other_vec))


if __name__ == "__main__":
    import numpy as np

    from color_perception_bench.colors import create_swatch_image

    sample_text = "red"
    text_embedding = get_text_embedding(sample_text)
    print(
        f"Text embedding for '{sample_text}':\n[{', '.join(map(str, text_embedding[:3]))}, ... ]\n"
    )

    sample_color = (255, 0, 0)  # red
    swatch = create_swatch_image(sample_color)
    image_embedding = get_image_embedding(swatch)
    print(
        f"Image embedding for RGB {sample_color}:\n[{', '.join(map(str, image_embedding[:3]))}, ... ]"
    )

    # print norm of each vector:
    img_vec_norm = np.linalg.norm(image_embedding)
    text_vec_norm = np.linalg.norm(text_embedding)
    print(f"\nImage embedding norm: {img_vec_norm}")
    print(f"Text embedding norm: {text_vec_norm}")
    print(
        f"Cosine distance between text and image embeddings: {cosine_distance(text_embedding, image_embedding)}"
    )
