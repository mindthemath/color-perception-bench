# Integration Plan: ColPali-v1.2

## 1. Model Overview
- **Name:** `vidore/colpali-v1.2`
- **Release Date:** June 27, 2024
- **Developer:** Vidore / Hugging Face contributors.
- **Type:** Late Interaction (ColBERT architecture) on top of PaliGemma.
- **Architecture:** 
  - **Vision:** SigLIP (finetuned).
  - **Language:** PaliGemma-3B.
  - **Mechanism:** "Late Interaction" - No single vector. The model outputs a *bag of vectors* (one for each image patch and text token).
- **Core Concept:** Relevance is computed by `Sum(Max(Sim(query_token, image_patch)))`.

## 2. Suitability for Visual Retrieval
- **Verdict:** **Very High (Quality) but Architecturally Incompatible**.
- **Reasoning:**
  - ColPali is currently SOTA for document/visual retrieval because it preserves spatial details that dense embeddings (like CLIP) compress and lose.
  - However, it **does not output a single embedding**.

## 3. Integration Feasibility

### A. Architectural Mismatch
Our current benchmark (`benchmark.py`) relies on:
1. `get_text_embedding() -> vector`
2. `get_image_embedding() -> vector`
3. `cosine_similarity(vector, vector)`

ColPali **breaks this abstraction**.
- It requires `score(text, image)` directly.
- It cannot be cached effectively as a single vector (requires caching matrices).

### B. Docker Service
- **Feasibility:** **Medium**.
- **Requirements:** `transformers >= 4.44`, `torch`.
- **Size:** 3B model is lightweight (~6GB VRAM).

### C. Proposed Solution: "Scoring Provider"
We would need to extend `benchmark.py` to support a new provider type: `scorer`.
- Instead of fetching embeddings, we fetch **scores**.
- *Challenge:* This makes the $N \times N$ comparison matrix expensive ($O(N^2)$ inferences) if not optimized.
- *Workaround:* We can extract the "multivectors" (matrices) and perform the ColBERT operation locally in Python, but the storage/RAM requirements are much higher (1024 vectors per image vs 1).

## 4. Implementation Details
**Docker Service API:**
- `POST /score`: `{"text": "...", "image": "..."}` -> `{"score": 0.85}`
- `POST /embed/multivector`: `{"image": "..."}` -> `{"embeddings": [[...], [...]]}` (Tensor output).

## 5. Recommendation
**High Priority for Future-Proofing.**
This represents the next generation of retrieval.
- **Immediate Action:** Do not integrate as a standard "embedding model".
- **Plan:** Create a separate branch or specialized "Late Interaction" benchmark mode.
- **For now:** Mark as "Requires Architecture Update".
