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