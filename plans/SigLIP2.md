# Integration Plan: Google SigLIP 2

## 1. Model Overview
- **Name:** `google/siglip2-so400m-patch14-384` (and potential larger variants like `so400m-patch14-384`)
- **Type:** Vision-Text Dual Encoder (CLIP-style)
- **Release Date:** February 2025
- **Developer:** Google

## 2. Why this model?
- **Architecture:** SigLIP (Sigmoid Loss for Language Image Pre-training) 2 replaces the standard softmax loss of CLIP with a sigmoid loss. This allows it to scale better and handle label noise more effectively.
- **Performance:** SigLIP 2 is a strict upgrade over the original SigLIP and standard CLIP models, offering better zero-shot classification and retrieval without increasing inference cost (same ViT architecture).
- **Efficiency:** The `so400m` variant is highly efficient and fits easily within consumer GPU limits while outperforming larger legacy models.

## 3. Integration Strategy

### A. Docker Service (`src/siglip_service`)
We will create a lightweight Python service similar to `bge_service`.
- **Dependencies:** `torch`, `transformers`, `pillow`, `flask`/`fastapi`.
- **Base Image:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime` (or lighter equivalent).

### B. API Specification
Conform to the standard `local` provider schema:
- `POST /txt/embed`: Accepts `{"input": "text"}` -> Returns L2-normalized vector.
- `POST /img/embed`: Accepts `{"input": "base64_image"}` -> Returns L2-normalized vector.

### C. Implementation Nuances
- **Pooling:** SigLIP typically uses an attention pooling mechanism or specific token extraction. We must ensure we use the correct `pooler_output` or mean pooling as specified by the model card.
- **Normalization:** Unlike some CLIP variants, SigLIP embeddings might not be pre-normalized. We **must** L2-normalize them in the service before returning to ensure dot-product in `benchmark.py` equals cosine similarity.
- **Prompting:** SigLIP usually does not require special templates (like "A photo of..."), but we should verify if SigLIP 2 introduced any prefix requirements.

## 4. Benchmark Configuration
Add to `models.yaml`:
```yaml
  siglip2-so400m:
    provider_type: local
    base_url: http://localhost:8002  # Assign distinct port
    text_endpoint:
      path: /txt/embed
    image_endpoint:
      path: /img/embed
```
