# Integration Plan: Qwen3-VL-Embedding

## 1. Model Overview
- **Name:** `Qwen/Qwen3-VL-Embedding-8B` (or `2B` variant for faster iteration)
- **Type:** Multimodal Embedding Model (Dual-Tower / Unified)
- **Release Date:** Late 2025 (Part of Qwen3-VL suite)
- **Developer:** Alibaba Cloud / Qwen Team

## 2. Why this model?
- **SOTA Performance:** It currently holds top rankings on the MMEB (Massive Multimodal Embedding Benchmark).
- **True Embedding Model:** Unlike `Qwen3-VL-Instruct`, this model is explicitly trained to output dense vectors for retrieval. We do not need to hack hidden states or guess pooling strategies.
- **Resolution Support:** Handles dynamic resolution and aspect ratios better than standard fixed-size CLIP models.

## 3. Integration Strategy

### A. Docker Service (`src/qwen_embedding_service`)
This requires a heavier environment than SigLIP due to the Qwen architecture.
- **Dependencies:** `torch`, `transformers>=4.45`, `qwen-vl-utils`, `flash-attn` (optional but recommended), `accelerate`.
- **Hardware:** The 8B model will require ~16-24GB VRAM. The 2B model fits comfortably on consumer cards (<8GB).

### B. API Specification
Conform to the standard `local` provider schema:
- `POST /txt/embed`: Accepts `{"input": "text"}`.
- `POST /img/embed`: Accepts `{"input": "base64_image"}`.

### C. Implementation Nuances
- **Inputs:** Requires `qwen-vl-utils` to process vision info (`process_vision_info`).
- **Prompting:** The model might expect specific role tags even for embedding tasks (e.g., `<|im_start|>user...`), or it might expose a `.encode()` method if using specific sentence-transformer wrappers. We will likely use the raw `transformers` implementation:
  - Text: `model.get_text_features(**inputs)`
  - Image: `model.get_image_features(**inputs)`
- **Dimension:** Verify output dimension (likely 3584 for 8B or 1536/similar for 2B).
- **Matryoshka:** If the model supports Matryoshka embeddings, we should default to the full dimension for maximum accuracy, or expose a config option to truncate.

## 4. Benchmark Configuration
Add to `models.yaml`:
```yaml
  qwen3-vl-embed-8b:
    provider_type: local
    base_url: http://localhost:8003
    text_endpoint:
      path: /txt/embed
    image_endpoint:
      path: /img/embed
```
