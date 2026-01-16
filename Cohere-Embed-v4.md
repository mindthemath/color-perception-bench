# Implementation Plan: Cohere Embed v4

## Goal
Integrate Cohere Embed v4 as a new multimodal embedding provider within the `color-perception-bench` project. This will enable the system to generate embeddings for text and images using the Cohere API.

## High-Level Steps

1.  **API Access and Authentication:**
    *   Obtain a Cohere API key from the Cohere dashboard.
    *   The API key is typically passed in the Authorization header for requests.

2.  **Python Client Setup:**
    *   Use the official Cohere Python SDK (`cohere` package). This simplifies API interaction.

3.  **Create New Provider Module:**
    *   Create a new Python file: `src/color_perception_bench/providers/cohere.py`.
    *   Define a class, e.g., `CohereEmbeddingProvider`, that inherits from `src/color_perception_bench/providers/base.py.EmbeddingProvider`.
    *   Implement the `embed_text(self, texts: list[str]) -> list[list[float]]` and `embed_image(self, images: list[Image.Image]) -> list[list[float]]` methods. Cohere Embed v4 supports embedding text and single images, so these can be implemented by making separate calls to the Cohere API. The image input will likely need to be converted to base64 or a similar format for the API.

4.  **Configuration:**
    *   Add configuration for Cohere Embed v4 to `models.yaml` (e.g., `cohere_api_key`, `cohere_model_name`).
    *   Ensure the provider can load these configurations securely, preferably via environment variables which override `models.yaml`.

5.  **Integrate with `embeddings.py`:**
    *   Add a new entry, `COHERE_EMBED_V4`, to the `EmbeddingModel` enum in `src/color_perception_bench/embeddings.py`.
    *   Modify the `create_embedding_model` function in `src/color_perception_bench/embeddings.py` to instantiate `CohereEmbeddingProvider` when `EmbeddingModel.COHERE_EMBED_V4` is selected.

6.  **Manage Dependencies:**
    *   Add `cohere` to the `requirements.txt` file within the `src/color_perception_bench/providers/` directory (or wherever other provider-specific dependencies are managed).
    *   Ensure these dependencies are reflected in the project's root `pyproject.toml` or `uv.lock` if managed by `uv`.

7.  **Testing:**
    *   Add unit tests for `CohereEmbeddingProvider` to verify correct API interaction and embedding generation.
    *   Consider adding an end-to-end test case using the `benchmark.py` or `experiment.py` framework to confirm functional integration.

## Example Code Snippet (Conceptual)

```python
# src/color_perception_bench/providers/cohere.py
from typing import List
from PIL import Image
import cohere
import base64
from io import BytesIO

from .base import EmbeddingProvider

class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str = "embed-multilingual-v3.0"): # Or "embed-english-v3.0", "embed-multimodal"
        self.co = cohere.Client(api_key)
        self.model_name = model_name

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        response = self.co.embed(
            texts=texts,
            model=self.model_name,
            input_type="classification" # or "search_query", "search_document", "clustering"
        )
        return response.embeddings

    def embed_image(self, images: List[Image.Image]) -> List[List[float]]:
        # Cohere API expects image inputs as base64 encoded strings
        base64_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG") # or JPEG, depending on API requirements
            base64_images.append(base64.b64encode(buffered.getvalue()).decode())

        response = self.co.embed(
            texts=[], # No text for image-only embedding
            images=base64_images,
            model="embed-multimodal", # Cohere's dedicated multimodal embedding model
            input_type="classification"
        )
        return response.embeddings

# src/color_perception_bench/embeddings.py (modifications)
# ...
class EmbeddingModel(Enum):
    # ... existing models
    COHERE_EMBED_V4 = "cohere-embed-v4"

def create_embedding_model(model_name: EmbeddingModel, config: Dict) -> EmbeddingProvider:
    # ... existing logic
    if model_name == EmbeddingModel.COHERE_EMBED_V4:
        api_key = config.get("cohere_api_key")
        if not api_key:
            raise ValueError("Cohere API key must be provided for Cohere Embed v4.")
        model_name_config = config.get("cohere_model_name", "embed-multilingual-v3.0") # default
        return CohereEmbeddingProvider(api_key=api_key, model_name=model_name_config)
    # ...
```