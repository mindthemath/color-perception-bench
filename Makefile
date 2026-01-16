run:
	uv run color-perception-bench

exp: lint
	uv run src/color_perception_bench/experiment.py

lint:
	uvx black src
	uvx isort --profile black src
	uvx ruff check . --fix
	uvx ty check

test-embed:
	uv run src/color_perception_bench/embeddings.py

test-color:
	uv run src/color_perception_bench/colors.py

clean:
	find src -name "__pycache__" -exec rm -rf {} +
	find src -name "*.pyc" -exec rm -f {} +

# BGE Service
build-bge:
	docker build -t bge-service src/bge_service

run-bge:
	docker run -d --name bge-service -p 8001:8000 bge-service
	@echo "BGE Service running at http://localhost:8001"

stop-bge:
	docker stop bge-service || true
	docker rm bge-service || true

# SigLIP Service
build-siglip:
	docker build -t siglip-service src/siglip_service

run-siglip:
	docker run -d --name siglip-service -p 8002:8002 siglip-service
	@echo "SigLIP Service running at http://localhost:8002"

stop-siglip:
	docker stop siglip-service || true
	docker rm siglip-service || true

# Qwen3-VL-Embedding Service
build-qwen-embed:
	docker build -t qwen-embed-service src/qwen_embedding_service

run-qwen-embed:
	docker run -d --name qwen-embed-service -p 8003:8000 --gpus all qwen-embed-service
	@echo "Qwen3-VL-Embedding service running at http://localhost:8003"

stop-qwen-embed:
	docker stop qwen-embed-service || true
	docker rm qwen-embed-service || true

test-qwen-embed:
	@python - <<'PY'
	import base64
	import io
	import json
	from PIL import Image
	import requests

	base_url = "http://localhost:8003"
	health = requests.get(f"{base_url}/health", timeout=10)
	health.raise_for_status()
	print("health:", health.json())

	payload = {"input": "red"}
	resp = requests.post(f"{base_url}/txt/embed", json=payload, timeout=60)
	resp.raise_for_status()
	txt = resp.json()["embedding"]
	print("text embedding dim:", len(txt))

	img = Image.new("RGB", (64, 64), (255, 0, 0))
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
	payload = {"input": f"data:image/png;base64,{img_b64}"}
	resp = requests.post(f"{base_url}/img/embed", json=payload, timeout=60)
	resp.raise_for_status()
	img_emb = resp.json()["embedding"]
	print("image embedding dim:", len(img_emb))
	PY
