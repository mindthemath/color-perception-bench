run:
	uv run color-perception-bench

exp: lint
	uv run src/color_perception_bench/experiment.py

lint:
	uvx black src
	uvx isort --profile black src
	uvx ty check
	uvx ruff check .

test-embed:
	uv run src/color_perception_bench/embeddings.py

test-color:
	uv run src/color_perception_bench/colors.py

clean:
	find src -name "__pycache__" -exec rm -rf {} +
	find src -name "*.pyc" -exec rm -f {} +
