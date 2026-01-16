from transformers import AutoModel, AutoProcessor
import os

MODEL_NAME = "google/siglip2-so400m-patch14-384"
CACHE_DIR = "/app/model_cache"

print(f"Downloading {MODEL_NAME} to {CACHE_DIR}...")
try:
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
    print("Download complete.")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
