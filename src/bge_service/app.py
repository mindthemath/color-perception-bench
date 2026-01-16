import base64
import io
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel

app = FastAPI()

MODEL_NAME = "BAAI/bge-vl-base"

print(f"Loading model {MODEL_NAME}...")
# Load model and processor
# bge-vl-base uses a CLIP-like architecture
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
try:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception:
    # CLIP-based models often use the CLIPProcessor
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()
print("Model loaded successfully.")


class TextRequest(BaseModel):
    input: str


class ImageRequest(BaseModel):
    input: str  # Base64 encoded image


class EmbeddingResponse(BaseModel):
    embedding: List[float]


@app.post("/txt/embed", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    try:
        with torch.no_grad():
            inputs = processor(
                text=request.input, return_tensors="pt", padding=True, truncation=True
            )
            # Use get_text_features for CLIP-like models
            if hasattr(model, "get_text_features"):
                embedding = model.get_text_features(**inputs)
            else:
                outputs = model(**inputs)
                embedding = (
                    outputs.text_embeds
                    if hasattr(outputs, "text_embeds")
                    else outputs[0]
                )

            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/img/embed", response_model=EmbeddingResponse)
async def embed_image(request: ImageRequest):
    try:
        # Decode base64 image (handle data URI scheme if present)
        input_str = request.input
        if input_str.startswith("data:"):
            input_str = input_str.split(",", 1)[1]

        image_data = base64.b64decode(input_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")

            if hasattr(model, "get_image_features"):
                embedding = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                embedding = (
                    outputs.image_embeds
                    if hasattr(outputs, "image_embeds")
                    else outputs[0]
                )

            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
