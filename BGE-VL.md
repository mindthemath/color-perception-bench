# Integration Plan: BAAI BGE-VL

## 1. Model Overview
- **Name:** `BAAI/bge-visualized-base-en-v1.5` (and variants like `bge-visualized-m3`)
- **Developer:** BAAI (Beijing Academy of Artificial Intelligence)
- **Type:** Dense Multimodal Embedding
- **Architecture:** CLIP-style dual encoder (Visual Encoder + Text Encoder) but trained with specialized "Visualized" data to align fine-grained visual features with text.
- **License:** MIT or Apache 2.0 (Check specific model card, usually permissive).

## 2. Suitability for Visual Retrieval
- **Verdict:** **High**.
- **Reasoning:**
  - Explicitly designed for "Visualized Information Retrieval" (VisIR).
  - Produces fixed-size embeddings (unlike ColPali), making it a drop-in replacement for `nomic` or `clip`.
  - Claims state-of-the-art performance on zero-shot classifications and retrieval benchmarks (MMEB).

## 3. Integration Feasibility

### A. Self-Hosting (Docker)
- **Feasibility:** **High**.
- **Dependencies:** `torch`, `transformers`, `pillow`.
- **Hardware:** Standard GPU support. 
- **VRAM:** 
  - `bge-visualized-base-en-v1.5` is relatively small (~300M parameters for ViT-L + BERT base). 
  - Easily runs on consumer GPUs (even CPU for inference is viable).

### B. API Availability
- **Public API:** Not widely available as a commercial API (unlike Voyage/Jina). 
- **Recommendation:** Self-host via Docker.

## 4. Implementation Details

### Docker Service (`src/bge_service`)
We can create a lightweight FastAPI service similar to the `local` provider pattern.

**Code Snippet (Concept):**
```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# Load model
model = AutoModel.from_pretrained('BAAI/bge-visualized-base-en-v1.5', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-visualized-base-en-v1.5')

def get_embedding(text=None, image=None):
    with torch.no_grad():
        if image:
            # Note: Specific preprocessing might be needed based on model card
            inputs = tokenizer(images=image, return_tensors="pt")
            emb = model.get_image_features(**inputs)
        elif text:
            inputs = tokenizer(text=text, return_tensors="pt")
            emb = model.get_text_features(**inputs)
    return emb.tolist()[0]
```

### Challenges / Nuances
- **Pre-processing:** Ensure the correct image transform is used (standard CLIP transforms vs BGE specific).
- **Weights:** Some BGE models are just weights for CLIP architecture; others use custom `Visualized_BGE` classes. Need to verify `trust_remote_code=True`.

## 5. Recommendation
Prioritize this integration. It fits the existing `models.yaml` schema perfectly (single vector output) and represents a strong open-source baseline.
