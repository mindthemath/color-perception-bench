import base64
import io
import os
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-Embedding-2B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading model {MODEL_NAME} on {DEVICE}...", flush=True)
model = AutoModel.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.", flush=True)


class TextRequest(BaseModel):
    input: str | List[str]


class ImageRequest(BaseModel):
    input: str | List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[float] | List[List[float]]


def _to_device(inputs: dict) -> dict:
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _extract_embedding(outputs, kind: str) -> torch.Tensor:
    if kind == "text":
        if hasattr(outputs, "text_embeds"):
            return outputs.text_embeds
    if kind == "image":
        if hasattr(outputs, "image_embeds"):
            return outputs.image_embeds
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    return outputs[0]


def _prepare_image_inputs(images: List[Image.Image]) -> dict:
    try:
        return processor(images=images, return_tensors="pt")
    except Exception as exc:
        if process_vision_info is None:
            raise exc
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
            for image in images
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[" "] * len(images),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)


def _decode_images(payload: str | List[str]) -> List[Image.Image]:
    if isinstance(payload, str):
        payloads = [payload]
    else:
        payloads = payload

    images = []
    for entry in payloads:
        if entry.startswith("data:"):
            entry = entry.split(",", 1)[1]
        image_data = base64.b64decode(entry)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        images.append(image)
    return images


@app.post("/txt/embed", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        inputs = processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = _to_device(inputs)

        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                embedding = model.get_text_features(**inputs)
            else:
                outputs = model(**inputs)
                embedding = _extract_embedding(outputs, "text")
            embedding = _normalize(embedding)

        if isinstance(request.input, list):
            return {"embedding": [emb.tolist() for emb in embedding]}
        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/img/embed", response_model=EmbeddingResponse)
async def embed_image(request: ImageRequest):
    try:
        images = _decode_images(request.input)
        inputs = _prepare_image_inputs(images)
        inputs = _to_device(inputs)

        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                embedding = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                embedding = _extract_embedding(outputs, "image")
            embedding = _normalize(embedding)

        if isinstance(request.input, list):
            return {"embedding": [emb.tolist() for emb in embedding]}
        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
