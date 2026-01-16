# Integration: BAAI BGE-VL

## 1. Finalized Model
- **Name:** `BAAI/bge-vl-base`
- **Type:** CLIP-based Multimodal Embedding.
- **Why this model?** 
  - `BAAI/bge-visualized-base-en-v1.5` was found to be gated/private on Hugging Face (returned 401).
  - `BAAI/bge-vl-base` is publicly accessible and provides a robust multimodal baseline.

## 2. Integration Status
- **Docker Service:** Implemented in `src/bge_service`.
- **API Spec:** Conforms to the `local` provider schema:
  - `POST /txt/embed`
  - `POST /img/embed`
- **Port:** Hosted on `http://localhost:8001`.

## 3. Usage
Build and run the service using the provided Makefile:
```bash
make build-bge
make run-bge
```

Add to `models.yaml`:
```yaml
  bge-vl-base:
    provider_type: local
    base_url: http://localhost:8001
    text_endpoint:
      path: /txt/embed
      method: POST
      input_field: input
      output_field: embedding
    image_endpoint:
      path: /img/embed
      method: POST
      input_field: input
      output_field: embedding
    user_batch_size: 1
```

## 4. Nuances
- **Architecture:** Uses `MMRet_CLIP` (custom modeling script downloaded automatically from HF).
- **Processing:** Uses `CLIPProcessor` (or `AutoProcessor` if compatible).
- **Normalization:** Embeddings are L2-normalized in the service layer for direct cosine distance compatibility.
