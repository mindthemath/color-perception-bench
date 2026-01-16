# Integration Plan: Qwen3-VL-4B-Instruct

## Objective
Integrate `Qwen/Qwen3-VL-4B-Instruct` into the `color-perception-bench` as a local provider for semantic search retrieval tasks (text-to-image alignment).

## 1. Architectural Approach
Since `Qwen3-VL-4B-Instruct` is a **generative** model (predicts next tokens) and not natively an **embedding** model (like CLIP or `Qwen3-VL-Embedding`), we cannot use it "as-is" via a simple `.encode()` API.

**Strategy:** We will treat the model's **last hidden state** (the internal "thought" vector before it generates text) as the embedding.
- **For Text:** Feed the text prompt and extract the vector of the last token (EOS).
- **For Images:** Feed the image (wrapped in special tokens) and extract the mean pooled vector of the image tokens, or the last token vector.

## 2. Docker Service (`src/qwen_service`)
To ensure portability and isolate heavy dependencies (Flash Attention 2), we will create a dedicated Docker service.

### `Dockerfile`
- **Base Image:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime` (or compatible).
- **System Deps:** `git` (for installing bleeding-edge transformers).
- **Python Deps:**
  - `git+https://github.com/huggingface/transformers` (Required for Qwen2.5/3 VL support).
  - `qwen-vl-utils` (Official helper library for processing inputs).
  - `accelerate` (For device map handling).
  - `flash-attn` (Critical for performance/memory).
  - `fastapi`, `uvicorn` (For the HTTP API).

### `app.py` (The Service)
A FastAPI server that exposes two endpoints compatible with our `models.yaml` `local` provider schema:
- `POST /txt/embed`: Accepts `{"input": "text"}` -> Returns `{"embedding": [...]}`.
- `POST /img/embed`: Accepts `{"content": "base64_string"}` -> Returns `{"embedding": [...]}`.

**Implementation Details:**
- **Loading:** `Qwen2VLForConditionalGeneration` (or Qwen3 specific class if available) with `device_map="auto"` and `torch_dtype="auto"`.
- **Extraction Logic:**
  - Run `model(..., output_hidden_states=True)`.
  - Access `outputs.hidden_states[-1]` (the last layer).
  - Perform **Mean Pooling** or **Last Token Extraction** to get a 1D vector.

## 3. Qwen3-VL Specific Nuances
- **Dynamic Resolution:** Qwen3-VL supports "Min-pixels" and "Max-pixels".
  - *Plan:* We will set a reasonable `max_pixels` (e.g., 1024x1024) to prevent OOM on large images while maintaining detail.
- **Video Support:** The model supports video, but we will strictly configure it for single-image input to optimize latency.
- **Prompt Template:** The model expects a chat format (`<|im_start|>...`).
  - *Plan:* We will construct raw inputs using `process_vision_info` from `qwen-vl-utils` to bypass conversational fluff and get raw features.

## 4. Integration Steps
1.  **Create Service:**
    - `src/qwen_service/Dockerfile`
    - `src/qwen_service/app.py`
    - `src/qwen_service/requirements.txt`
2.  **Orchestration:**
    - Add `qwen3-vl` service to a new `docker-compose.yml`.
    - Map port `8001` (host) to `8000` (container).
3.  **Configuration:**
    - Add entry to `models.yaml`:
      ```yaml
      qwen3-vl-4b:
        provider_type: local
        base_url: http://localhost:8001
        text_endpoint: ...
        image_endpoint: ...
      ```
4.  **Validation:**
    - Verify embedding dimensions (likely 3584 or similar depending on the 4B architecture).
    - Ensure cosine similarity between "red" (text) and Red Swatch (image) is positive.
