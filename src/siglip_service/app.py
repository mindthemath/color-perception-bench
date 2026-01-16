import base64
import io
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

app = FastAPI()

MODEL_NAME = "google/siglip2-so400m-patch14-384"

print(f"Loading model {MODEL_NAME}...", flush=True)
try:
    # Trust remote code is sometimes needed for newer models
    # The model should be pre-cached in the Docker image at HF_HOME
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    raise e

model.eval()
print("Model loaded successfully.", flush=True)


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
            # SigLIP processor usually handles text tokenization
            inputs = processor(
                text=request.input, return_tensors="pt", padding="max_length", truncation=True
            )
            
            # Remove 'pixel_values' if present in text-only inputs (AutoProcessor might add them if not careful, 
            # but usually 'text=' argument is sufficient)
            
            if hasattr(model, "get_text_features"):
                embedding = model.get_text_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, "text_embeds"):
                    embedding = outputs.text_embeds
                elif hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output
                else:
                    # Fallback: use first token or mean pooling if not explicit
                    # This depends heavily on architecture, but SigLIP usually has text_embeds
                    embedding = outputs[0]

            # EXPLICIT L2 Normalization as per instructions
            embedding = F.normalize(embedding, p=2, dim=1)

        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/img/embed", response_model=EmbeddingResponse)
async def embed_image(request: ImageRequest):
    try:
        # Decode base64 image
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
                if hasattr(outputs, "image_embeds"):
                    embedding = outputs.image_embeds
                elif hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs[0]

            # EXPLICIT L2 Normalization as per instructions
            embedding = F.normalize(embedding, p=2, dim=1)

        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
